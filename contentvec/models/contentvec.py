# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..data.data_utils import compute_mask_indices, lengths_to_padding_mask
from ..modules.utils import buffered_arange, index_put, is_xla_tensor
from .wav2vec2_1 import ConvFeatureExtractionModel, TransformerEncoder_1
from ..modules.norms import LayerNorm


logger = logging.getLogger(__name__)

@dataclass
class ContentvecConfig:
    _name: str
    label_rate: int
    extractor_mode: str
    encoder_layers: int
    encoder_layers_1: int
    encoder_embed_dim: int
    encoder_ffn_embed_dim: int
    encoder_attention_heads: int
    activation_fn: str
    ctr_layers: List[int]
    dropout: float
    attention_dropout: float
    activation_dropout: float
    encoder_layerdrop: float
    dropout_input: float
    dropout_features: float
    final_dim: int
    untie_final_proj: bool
    layer_norm_first: bool
    conv_feature_layers: str
    conv_bias: bool
    logit_temp: float
    logit_temp_ctr: float
    target_glu: bool
    feature_grad_mult: float
    mask_length: int
    mask_prob: float
    mask_selection: str
    mask_other: float
    no_mask_overlap: bool
    mask_min_space: int
    mask_channel_length: int
    mask_channel_prob: float
    mask_channel_selection: str
    mask_channel_other: float
    no_mask_channel_overlap: bool
    mask_channel_min_space: int
    num_negatives: int
    cross_sample_negatives: int
    conv_pos: int
    conv_pos_groups: int
    latent_temp: List[float]
    skip_masked: bool
    skip_nomask: bool

class ContentvecModel(nn.Module):
    def __init__(self, cfg: ContentvecConfig) -> None:
        super().__init__()
        logger.info(f"ContentvecModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        #####################
        self.sample_rate = 16000
        #####################
        self.feat2tar_ratio = (
            cfg.label_rate * feature_ds_rate / self.sample_rate
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.logit_temp_ctr = cfg.logit_temp_ctr
        self.ctr_layers = cfg.ctr_layers
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        
        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.num_updates = 0

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder_1(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

            
        self.layer_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def get_mask(self, B, T, padding_mask):
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            #mask_indices = torch.from_numpy(mask_indices).to(x.device)
            #x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        assert self.mask_channel_prob == 0

        return mask_indices
    
    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs
    
    def compute_sim(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / self.logit_temp_ctr

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor, 
                         padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.feature_extractor(source, padding_mask)
        return features

    def forward_targets(
        self, features: torch.Tensor, target_list: List[torch.Tensor], padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            padding_mask = padding_mask[:, :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list, padding_mask

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # replaces original forward_padding_mask for batch inference
        lengths_org = (~padding_mask).long().sum(dim=1)
        lengths = (lengths_org - 400).div(320, rounding_mode='floor') + 1
        padding_mask = lengths_to_padding_mask(lengths)
        return padding_mask
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self,
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        spk_emb: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask_1: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        tap: bool = True
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        
        source = torch.cat((source_1, source_2), dim=0)
        padding_mask_2 = padding_mask_1.clone()
        padding_mask = torch.cat((padding_mask_1, padding_mask_2), dim=0)
        spk_emb = spk_emb.repeat(2, 1)
        
        features = self.forward_features(source, padding_mask)
        
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        
        if target_list is not None:
            features, target_list, padding_mask = self.forward_targets(features, target_list, padding_mask)
            for j,t in enumerate(target_list):
                target_list[j] = t.repeat(2,1)
                
        features_pen = features.float().pow(2).mean()
        
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        #unmasked_features = features.clone()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        #unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            B, T, _ = features.shape
            mask_indices = self.get_mask(B//2, T, padding_mask_1)
            mask_indices = torch.from_numpy(mask_indices).to(features.device)
            mask_indices = mask_indices.repeat(2, 1)
            features[mask_indices] = self.mask_emb
            x = features
            #unmasked_indices = torch.logical_and(~padding_mask, ~mask_indices)
            unmasked_indices = ~mask_indices
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            spk_emb,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
            tap=tap
        )
        
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}
        
        # prepare contrastive loss
        score_list = []
        for ctr_layer in self.ctr_layers:
            y = layer_results[max(ctr_layer, -len(layer_results))]   # LAYER DROP??
            y = y[unmasked_indices].view(y.size(0), -1, y.size(-1))
            y_1, y_2 = torch.split(y, B//2, dim=0)
            y_1 = self.layer_proj(y_1)
            y_2 = self.layer_proj(y_2)
            
            negs_1, _ = self.sample_negatives(y_1, y_1.size(1))
            negs_2, _ = self.sample_negatives(y_2, y_2.size(1))
            z_1 = self.compute_sim(y_1, y_2, negs_1)
            z_2 = self.compute_sim(y_2, y_1, negs_2)
            z = torch.cat((z_1, z_2), dim=1)
            score_list.append(z)
        
        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices) 
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(
                    zip(proj_x_m_list, target_list)
                )
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(
                    zip(proj_x_u_list, target_list)
                )
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            "score_list": score_list
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        spk_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        tap: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:      
        
        features = self.forward_features(source, padding_mask)
        
        if padding_mask is not None:
            logger.info("Batch generation mode!")
            padding_mask = self.forward_padding_mask(features, padding_mask)
        
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x, layer_results = self.encoder(
            features,
            spk_emb,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
            tap=tap
        )
        res = features if ret_conv else x
        
        return res, padding_mask

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [
            x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
        ]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        
    def get_logits_ctr(self, net_output):
        logits_list = net_output["score_list"]
        if len(logits_list) > 1:
            logits_B = []
            for logits in logits_list:
                logits = logits.transpose(0, 2)
                logits = logits.reshape(-1, logits.size(-1))
                logits_B.append(logits)
            logits_B = torch.cat(logits_B, dim=0)
        else:
            logits = logits_list[0]
            logits = logits.transpose(0, 2)
            logits_B = logits.reshape(-1, logits.size(-1))
        return logits_B

    def get_targets_ctr(self, net_output):
        logits_list = net_output["score_list"]
        logits = logits_list[0]
        return logits.new_zeros(
            logits.size(1) * logits.size(2) * len(logits_list), 
            dtype=torch.long)
