import torch
from transformers import AutoTokenizer
from models import CCSRD   

def tokenize(tokenizer, texts):
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=10
        )
        return encoded["input_ids"]



# -------------------
# INFERENCE
# -------------------

def infer(model, waveform, tokenizer, device, max_len=20):

    model.eval()
    with torch.no_grad():
        waveform = waveform.unsqueeze(0).to(device)
        H   = model.speech_encoder(waveform)
        Z_c = model.content_encoder(H)

        # start with BOS token
        generated = torch.tensor([[tokenizer.bos_token_id or 0]],
                                  dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = model.tgt_embedding(generated)
            logits  = model.translation_decoder(tgt_emb, Z_c)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(generated[0].cpu().tolist(),
                                skip_special_tokens=True)