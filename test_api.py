import torch
import torchaudio

from contentvec.api import ContentvecWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Inference on {device}:')
            
vec_path = './checkpoint_best_legacy_500_converted.pt'
vec_model = ContentvecWrapper(vec_path).to(device)

audio, sample_rate = torchaudio.load('test.wav')
feature = vec_model(audio, sample_rate)

print(f'feature shape: {feature.shape}')

audio_length = audio.size(-1) / sample_rate
batch_size, feature_length, hidden_dim = feature.shape

print(f'frame_rate: {feature_length / audio_length}')