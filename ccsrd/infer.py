# infer.py

import torch
from transformers import AutoTokenizer

from models import CCSRD
from data import load_data_samples, build_speaker_map, ExpressoDataset, collate
from utils import infer, tokenize


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load checkpoint
checkpoint = torch.load("checkpoint_low_epoch.pt", map_location=device, weights_only=False)

spk2id = checkpoint["spk2id"]
train_samples = checkpoint["train_samples"] 
num_speakers = len(spk2id)

# make sure a sample target is German not English
print(f"sample target: {train_samples[0]['tgt_text']['deu']}")

model = CCSRD(
    d_model=512,
    num_speakers=num_speakers,
    vocab_size=tokenizer.vocab_size
).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()


# --- teacher forcing check first ---
# this tells us if model learned ANYTHING at all
print("\n>>> TEACHER FORCING CHECK (should be near perfect)")
from data import ExpressoDataset, collate
from torch.utils.data import DataLoader

loader = DataLoader(
    ExpressoDataset(train_samples[:10], spk2id),
    batch_size=2, shuffle=False, collate_fn=collate
)

with torch.no_grad():
    waveform, texts, spk_labels = next(iter(loader))
    waveform   = waveform.to(device)
    tgt_tokens = tokenize(tokenizer, texts).to(device)

    outputs  = model(waveform, tgt_tokens=tgt_tokens, lam=1.0)
    pred_ids = torch.argmax(outputs["translation_logits"], dim=-1)

    for i in range(len(texts)):
        print(f"\n[{i}] TARGET : {texts[i]}")
        print(f"[{i}] TF PRED: {tokenizer.decode(pred_ids[i].cpu().tolist(), skip_special_tokens=True)}")


# --- autoregressive inference on training samples ---
print("\n>>> AUTOREGRESSIVE ON TRAINING SAMPLES")
for i in range(10):
    sample   = train_samples[i]
    waveform = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
    pred     = infer(model, waveform, tokenizer, device)

    print(f"\n[{i}] SRC   : {sample['src_text']}")
    print(f"[{i}] PRED  : {pred}")
    print(f"[{i}] TARGET: {sample['tgt_text']['deu']}")