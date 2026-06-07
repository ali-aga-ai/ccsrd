import torch
from transformers import AutoTokenizer
from models import CCSRD   
from transformers import AutoConfig

def tokenize(tokenizer, texts):
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        return encoded["input_ids"]



# -------------------
# INFERENCE
# -------------------
def infer(model, waveform, tokenizer, device, max_len=50):
    model.eval()
    with torch.no_grad():
        waveform = waveform.unsqueeze(0).to(device)
        H   = model.speech_encoder(waveform)
        Z_c = model.content_encoder(H)

        config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        start_id = config.decoder_start_token_id
        eos_id   = tokenizer.eos_token_id

        generated = torch.tensor([[start_id]], dtype=torch.long, device=device)

        for step in range(max_len):
            tgt_emb = model.tgt_embedding(generated)
            logits  = model.translation_decoder(tgt_emb, Z_c)

            # take last position only
            next_logits = logits[0, -1, :].clone().float()


            # stable top-p
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)

            sampled  = torch.multinomial(sorted_probs, 1)
            next_tok = sorted_idx[sampled].unsqueeze(0)  # (1, 1)

            generated = torch.cat([generated, next_tok], dim=1)

            if next_tok.item() == eos_id:
                break

        tokens = generated[0, 1:].cpu().tolist()
        if tokens and tokens[-1] == eos_id:
            tokens = tokens[:-1]

        return tokenizer.decode(tokens, skip_special_tokens=True)
