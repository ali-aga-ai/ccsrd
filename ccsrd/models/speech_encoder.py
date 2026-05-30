# speech_encoder.py

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model


class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec.requires_grad_(False)  # frozen, pretrained on LibriSpeech audio only

        # 768 -> 512, sequence length /4
        self.conv = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

    @staticmethod
    def load_audio(audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        return waveform  # (1, T)

    def forward(self, waveform):
        # waveform: (B, T_audio)  <-- caller passes tensor, not path
        H = self.wav2vec(waveform).last_hidden_state  # (B, T, 768)
        H = H.transpose(1, 2)                         # (B, 768, T)
        H = self.conv(H)                              # (B, 512, T/4)
        H = H.transpose(1, 2)                         # (B, T/4, 512)
        return H