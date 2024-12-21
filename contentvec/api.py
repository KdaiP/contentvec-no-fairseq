import torch
import torchaudio

from .models.contentvec import ContentvecConfig, ContentvecModel

class ContentvecWrapper(torch.nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        model_dict = torch.load(model_path, weights_only=True)
        
        self.config = ContentvecConfig(**model_dict['cfg']['model'])
        self.model = ContentvecModel(self.config)
        self.sample_rate = self.model.sample_rate # 16000
        
        self.model.load_state_dict(model_dict['model'], strict=False)
        self.model.eval()
        
    @ torch.inference_mode()
    def forward(self, audio: torch.Tensor, sample_rate: int, output_layer: int = 12) -> torch.Tensor:
        device = next(self.parameters()).device
        
        audio = audio[:1, ...] # convert to mono
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
            
        padding_mask = torch.BoolTensor(audio.shape).fill_(False)
        inputs = {
            "source": audio.to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": output_layer,
            "spk_emb": None
        }
        logits = self.model.extract_features(**inputs)[0]
        return logits # [batch_size, time, hidden_dim]