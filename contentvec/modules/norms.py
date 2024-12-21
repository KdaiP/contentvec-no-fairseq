# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    
class GroupNormMasked(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x, mask=None):
        B, C, L = x.size()
        assert C % self.num_groups == 0
        
        x = x.view(B, self.num_groups, C//self.num_groups, L)
        if mask is None:
            mask = torch.ones_like(x)
        else:
            mask = mask.view(B, 1, 1, L)
        x = x * mask
        lengths = mask.sum(dim=3, keepdim=True)
        
        assert x.size(2)==1
        mean_ = x.mean(dim=3, keepdim=True)
        mean = mean_ * L / lengths

        #var = (((x - mean)**2)*mask).sum(dim=3, keepdim=True) / lengths
        #var = (x**2).sum(dim=3, keepdim=True) / lengths - mean**2
        var = (x.var(dim=3, unbiased=False, keepdim=True) + mean_**2) * L / lengths - mean**2
        var = var.add_(self.eps)

        x = x.add_(-mean.detach())
        x = x.div_(var.sqrt().detach())
        
        x = x.view(B, C, L)
        
        x = x.mul_(self.weight.view(1,-1,1))
        x = x.add_(self.bias.view(1,-1,1))
        
        return x
    
    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
    
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    
class CondLayerNorm(nn.Module):

    def __init__(self, dim_last, eps=1e-5, dim_spk=256, elementwise_affine=True):
        super(CondLayerNorm, self).__init__()
        self.dim_last = dim_last
        self.eps = eps
        self.dim_spk = dim_spk
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight_ln = torch.nn.Linear(self.dim_spk, 
                                    self.dim_last, 
                                    bias=False)
            self.bias_ln = torch.nn.Linear(self.dim_spk, 
                                  self.dim_last, 
                                  bias=False)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight_ln.weight)
            torch.nn.init.zeros_(self.bias_ln.weight)

    def forward(self, input, spk_emb):
        weight = self.weight_ln(spk_emb)
        bias = self.bias_ln(spk_emb)
        return F.layer_norm(
            input, input.size()[1:], weight, bias, self.eps)

    def extra_repr(self):
        return '{dim_last}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
            
class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x
    
class CondLayerNorm(nn.Module):

    def __init__(self, dim_last, eps=1e-5, dim_spk=256, elementwise_affine=True):
        super(CondLayerNorm, self).__init__()
        self.dim_last = dim_last
        self.eps = eps
        self.dim_spk = dim_spk
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight_ln = nn.Linear(self.dim_spk, 
                                    self.dim_last, 
                                    bias=False)
            self.bias_ln = nn.Linear(self.dim_spk, 
                                  self.dim_last, 
                                  bias=False)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight_ln.weight)
            nn.init.zeros_(self.bias_ln.weight)

    def forward(self, input, spk_emb):
        weight = self.weight_ln(spk_emb)
        bias = self.bias_ln(spk_emb)
        return F.layer_norm(
            input, input.size()[1:], weight, bias, self.eps)

    def extra_repr(self):
        return '{dim_last}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
