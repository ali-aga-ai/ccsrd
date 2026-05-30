# train.py

import torch
import torch.optim as optim
from .ccsrd import CCSRD
from .utils  import CCSRDLoss
from .speech_encoder import SpeechEncoder


def train_ccsrd_example():
    d_model      = 512
    num_speakers = 100
    vocab_size   = 5000
    num_epochs   = 10
    lr           = 1e-4

    model   = CCSRD(d_model=d_model, num_speakers=num_speakers, vocab_size=vocab_size)
    loss_fn = CCSRDLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    for epoch in range(num_epochs):
        model.train()

        # --- load a real batch from your dataloader here ---
        # Dummy batch for illustration
        B       = 2
        T_audio = 16000 * 5          # 5 seconds at 16kHz
        T_tgt   = 20

        waveform    = torch.randn(B, T_audio)
        tgt_tokens  = torch.randint(0, vocab_size,   (B, T_tgt))
        spk_labels  = torch.randint(0, num_speakers, (B,))
        tgt_labels  = torch.randint(0, vocab_size,   (B, T_tgt))

        # anneal GRL lambda (common practice)
        lam = min(1.0, epoch / num_epochs * 2)

        outputs = model(waveform, tgt_tokens=tgt_tokens, lam=lam)

        loss_dict = loss_fn(outputs, spk_labels, tgt_labels)

        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"total={loss_dict['total']:.4f}  "
              f"L_CON={loss_dict['L_CON']:.4f}  "
              f"L_NCON={loss_dict['L_NCON']:.4f}  "
              f"L_REC={loss_dict['L_REC']:.4f}  "
              f"L_SPK={loss_dict['L_SPK']:.4f}  "
              f"L_ST={loss_dict['L_ST']:.4f}")


if __name__ == "__main__":
    train_ccsrd_example()