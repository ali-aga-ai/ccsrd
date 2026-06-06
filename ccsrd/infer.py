import torch
from transformers import AutoTokenizer

from models import CCSRD
from data import load_data_samples
from utils import infer 


# -------------------
# DEVICE + TOKENIZER
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("t5-small")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------
# LOAD MODEL
# -------------------
checkpoint = torch.load("checkpoint.pt", map_location=device)

spk2id = checkpoint["spk2id"]
num_speakers = len(spk2id)

model = CCSRD(
    d_model=512,
    num_speakers=num_speakers,
    vocab_size=tokenizer.vocab_size
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()


# -------------------
# LOAD SAMPLE DATA
# -------------------
small_ds = load_data_samples(num_samples=50)

# -------------------
# RUN INFERENCE
# -------------------
for i in range(10):
    sample = small_ds[i]
    waveform = torch.tensor(sample["audio"]["array"], dtype=torch.float32)

    print("\n====================")
    print("IDX:", i)
    print("SRC:", sample["src_text"])
    print("PRED:", infer(model, waveform, tokenizer, device))
    print("TARGET:", sample["tgt_text"]["deu"])