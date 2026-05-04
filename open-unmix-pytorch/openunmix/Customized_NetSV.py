from typing import List, Optional
import numpy as np
import torch

from scipy.spatial.distance import cdist

FFT_SIZE: int = 1024
HOP_LENGTH: int = 256
WINDOW = torch.hann_window(FFT_SIZE)

def stft(waveform: torch.Tensor):
    spectrogram = torch.stft(
        waveform, FFT_SIZE, HOP_LENGTH, window=WINDOW.to(waveform.device),
        return_complex=True  # change to True
    )
    spectrogram = torch.view_as_real(spectrogram)
    spectrogram = spectrogram.permute(0, 2, 1, 3)
    magnitude_spectrogram = torch.sqrt(spectrogram[..., 0] ** 2 + spectrogram[..., 1] ** 2)

    return spectrogram, magnitude_spectrogram

class NetSV(torch.nn.Module):

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.GRU(
            input_size=int(FFT_SIZE // 2 + 1),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

    def embedding(self, x):
        x = stft(x)[1]
        x = self.rnn(x)[0]
        return x[:, -1]

    def forward(self, x_1, x_2):
        feature_1 = self.embedding(x_1)
        feature_2 = self.embedding(x_2)
        is_same = torch.bmm(
            feature_1.unsqueeze(1),
            feature_2.unsqueeze(2)
        ).squeeze()
        return is_same