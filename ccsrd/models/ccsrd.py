# model.py

import torch
import torch.nn as nn

from .speech_encoder  import SpeechEncoder
from .encoders        import ContentEncoder, NonContentEncoder, TranslationDecoder
from .reconstructors  import CyclicReconstructor, FeatureReconstructor
from .classifier      import SpeakerClassifier


class CCSRD(nn.Module):
    def __init__(self, d_model=512, num_speakers=100, vocab_size=5000):
        super().__init__()
        self.d_model   = d_model
        self.vocab_size = vocab_size

        self.speech_encoder       = SpeechEncoder()
        self.content_encoder      = ContentEncoder(d_model)
        self.non_content_encoder  = NonContentEncoder(d_model)
        self.cyclic_reconstructor = CyclicReconstructor(d_model)
        self.feature_reconstructor= FeatureReconstructor(d_model)
        self.speaker_classifier   = SpeakerClassifier(d_model, num_speakers)
        self.translation_decoder  = TranslationDecoder(d_model, vocab_size)

        # token embedding for target-side tokens
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, waveform, tgt_tokens=None, lam=1.0):
        """
        waveform:   (B, T_audio)   raw waveform tensor
        tgt_tokens: (B, T_tgt)     integer token ids (optional, for training)
        lam:        GRL lambda scaling factor
        """
        # H is the speech encoder output. 
        H    = self.speech_encoder(waveform)       # (B, T, 512)
        Z_c  = self.content_encoder(H)             # (B, T, 512)
        Z_nc = self.non_content_encoder(H)         # (B, T, 512)

        z_c_pred, z_nc_pred = self.cyclic_reconstructor(Z_c, Z_nc, lam)
        H_recon             = self.feature_reconstructor(Z_c, Z_nc)
        speaker_logits      = self.speaker_classifier(Z_nc)

        translation_logits = None
        if tgt_tokens is not None:
            tgt_emb = self.tgt_embedding(tgt_tokens)           # (B, T_tgt, d_model)
            translation_logits = self.translation_decoder(tgt_emb, Z_c)  # (B, T_tgt, vocab)

        return {
            'H':                 H,
            'Z_c':               Z_c,
            'Z_nc':              Z_nc,
            'z_c_pred':          z_c_pred,
            'z_nc_pred':         z_nc_pred,
            'H_recon':           H_recon,
            'speaker_logits':    speaker_logits,
            'translation_logits':translation_logits,
        }

    # the following two functions are for inference.
    @torch.no_grad()
    def encode_audio(self, waveform):
        """Inference: only speech + content encoder needed."""
        H   = self.speech_encoder(waveform)
        Z_c = self.content_encoder(H)
        return Z_c

    @torch.no_grad()
    def translate(self, waveform, tgt_tokens):
        Z_c     = self.encode_audio(waveform)
        tgt_emb = self.tgt_embedding(tgt_tokens)
        return self.translation_decoder(tgt_emb, Z_c)