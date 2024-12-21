import torch
import torchaudio

from fairseq import checkpoint_utils
from contentvec.api import ContentvecWrapper

class FairseqContentvecWrapper(torch.nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
        self.model = models[0].eval()
        self.sample_rate = 16000
        
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
        }
        logits = self.model.extract_features(**inputs)[0]
        return logits # [batch_size, time, hidden_dim]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Inference on {device}:')

original_vec_path = './checkpoint_best_legacy_500.pt'            
conerted_vec_path = './checkpoint_best_legacy_500_converted.pt'
vec_model1 = ContentvecWrapper(conerted_vec_path).to(device)
vec_model2 = FairseqContentvecWrapper(original_vec_path).to(device)

audio, sample_rate = torchaudio.load('test.wav')
feature1 = vec_model1(audio, sample_rate)
feature2 = vec_model2(audio, sample_rate)

all_close = torch.allclose(feature1, feature2)

if all_close:
    print("Congratulations! The converted ContentVec model produces the same results as the original one.")
else:
    print("Mismatch! The converted ContentVec model produces different results from the original one.")