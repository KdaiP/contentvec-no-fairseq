import torch
from .contentvec.api import ContentvecWrapper

from vencoder.encoder import SpeechEncoder


class ContentVec768L12(SpeechEncoder):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        model = ContentvecWrapper(vec_path)
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = model.to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        logits = self.model.forward(wav, 16000)
        return logits.transpose(1, 2)
    
if __name__ == '__main__':
    import torchaudio
    
    audio, sample_rate = torchaudio.load('test.wav')
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        
    model = ContentVec768L12('checkpoint_best_legacy_500_converted.pt')
    feature = model.encoder(audio)
    print(feature.shape)