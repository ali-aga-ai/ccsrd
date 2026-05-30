# classifier.py

import torch.nn as nn


class SpeakerClassifier(nn.Module):
    """
    Predicts speaker ID from Z_nc (Eq. 7).
    Paper: FC layers + adaptive avg pool + log softmax.
    Loss: NLLLoss (paired with LogSoftmax here).
    """
    def __init__(self, d_model=512, num_speakers=100):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),      nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, num_speakers),
        )
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, z_nc):
        # z_nc: (B, T, d_model)
        pooled = self.pool(z_nc.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        logits = self.classifier(pooled)                       # (B, num_speakers)
        return self.log_softmax(logits)