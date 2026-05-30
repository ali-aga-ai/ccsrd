# reconstructors.py

import torch
import torch.nn as nn
from torch.autograd import Function


class GRL(Function):
    """Gradient Reversal Layer with scaling factor lambda."""
    @staticmethod
    def forward(ctx, x, lam=1.0):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None


class CyclicReconstructor(nn.Module):
    """
    Two separate predictors connected via GRL (Section 3.2, Eq. 4 & 5):
      - content_predictor:    predicts Z_c  from Z_nc  (L_CON)
      - noncontent_predictor: predicts Z_nc from Z_c   (L_NCON)
    GRL inverts gradients so encoders are pushed to maximise these losses,
    i.e. make content and non-content representations hard to reconstruct
    from each other.
    Architecture per paper: 3 FC+ReLU then 1 FC+Tanh.
    """
    def __init__(self, d_model=512):
        super().__init__()

        def _predictor():
            return nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.Tanh(),
            )

        self.content_predictor    = _predictor()  # predicts Z_c  from Z_nc
        self.noncontent_predictor = _predictor()  # predicts Z_nc from Z_c

    def forward(self, z_c, z_nc, lam=1.0):
        # Apply GRL: gradients are reversed when flowing back to the encoders
        z_c_grl  = GRL.apply(z_c,  lam)
        z_nc_grl = GRL.apply(z_nc, lam)

        z_c_pred  = self.content_predictor(z_nc_grl)    # (B, T, d_model)
        z_nc_pred = self.noncontent_predictor(z_c_grl)  # (B, T, d_model)

        return z_c_pred, z_nc_pred  # MSE against z_c / z_nc respectively


class FeatureReconstructor(nn.Module):
    """
    Reconstructs original speech H from Z_c ⊕ Z_nc (Eq. 6).
    No GRL -- we want both encoders to cooperate here.
    Architecture: same as predictor networks but input dim is d_model*2.
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),      nn.Tanh(),
        )

    def forward(self, z_c, z_nc):
        combined = torch.cat([z_c, z_nc], dim=-1)  # (B, T, d_model*2)
        return self.mlp(combined)                   # (B, T, d_model)