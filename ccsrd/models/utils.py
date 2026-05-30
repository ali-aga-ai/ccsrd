# loss.py

import torch.nn as nn


class CCSRDLoss(nn.Module):
    """
    Eq. 8:  L      = L_ST + L_SRD
    Eq. 9:  L_SRD  = L_CON + L_NCON + L_REC + L_SPK
    Eq. 10: L_MTL  = L_ST + L_ASR + L_MT + L_SRD
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super().__init__()
        self.alpha = alpha  # weight for L_SRD components
        self.delta = delta  # weight for translation loss
        self.mse = nn.MSELoss()
        self.nll = nn.NLLLoss()       # pairs with LogSoftmax in classifier
        self.ce  = nn.CrossEntropyLoss()  # for translation (applies softmax internally)

    def forward(self, outputs, speaker_labels, translation_labels):
        """
        outputs:           dict returned by CCSRD.forward()
        speaker_labels:    (B,)          integer speaker IDs
        translation_labels:(B, T_tgt)    integer token IDs
        """
        H    = outputs['H']
        Z_c  = outputs['Z_c']
        Z_nc = outputs['Z_nc']

        # Eq. 4 & 5 -- GRL already applied inside cyclic_reconstructor
        L_CON  = self.mse(outputs['z_c_pred'],  Z_c)
        L_NCON = self.mse(outputs['z_nc_pred'], Z_nc)

        # Eq. 6
        L_REC = self.mse(outputs['H_recon'], H)

        # Eq. 7
        L_SPK = self.nll(outputs['speaker_logits'], speaker_labels)

        L_SRD = L_CON + L_NCON + L_REC + L_SPK  # Eq. 9

        # Translation (Eq. 1)
        B, T_tgt, vocab_size = outputs['translation_logits'].shape
        L_ST = self.ce(
            outputs['translation_logits'].reshape(-1, vocab_size),
            translation_labels.reshape(-1)
        )

        total = self.delta * L_ST + self.alpha * L_SRD  # Eq. 8

        return {
            'total':  total,
            'L_CON':  L_CON,  'L_NCON': L_NCON,
            'L_REC':  L_REC,  'L_SPK':  L_SPK,
            'L_ST':   L_ST,
        }