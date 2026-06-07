import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer

from models import CCSRD, CCSRDLoss
from data import load_data_samples , build_speaker_map, collate, ExpressoDataset
from utils import tokenize


# -------------------
# DEVICE + TOKENIZER
# -------------------
NUM_EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# after loading tokenizer ()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



# -------------------
# small ds is a list of dicts, each with keys: 'audio', 'src_text', 'tgt_text', 'src_speaker'
# --------------------
small_ds = load_data_samples(num_samples = 3000) 

print(f"[INFO] Loaded {len(small_ds)} samples")

# ---------------
# giving each speaker an integer ID (0 to num_speakers-1)
# ---------------
spk2id, num_speakers = build_speaker_map(small_ds)


# -------------------
# DATALOADER
# -------------------
train_loader = DataLoader(
    ExpressoDataset(small_ds, spk2id),
    batch_size=2,
    shuffle=True,
    collate_fn=collate
)


# -------------------
# TRAINING
# -------------------

print("[INFO] Initializing model...")

model = CCSRD(
    d_model=512,
    num_speakers=num_speakers,
    vocab_size=tokenizer.vocab_size
).to(device)

loss_fn = CCSRDLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))

# pick one fixed sample at the start — never changes
PROBE_IDX = 0
probe_sample = small_ds[PROBE_IDX]
probe_wave   = torch.tensor(probe_sample["audio"]["array"], dtype=torch.float32).to(device)
probe_target = probe_sample["tgt_text"]["deu"]

def probe_epoch(model, epoch):
    """Runs one teacher-forced forward pass on the probe sample and prints diagnostics."""
    model.eval()
    with torch.no_grad():
        # tokenize target
        tgt_tokens = tokenize(tokenizer, [probe_target]).to(device)  # (1, T)
        tgt_labels = tgt_tokens.clone()
        tgt_labels[tgt_labels == tokenizer.pad_token_id] = -100

        outputs  = model(probe_wave.unsqueeze(0), tgt_tokens=tgt_tokens, lam=1.0)
        logits   = outputs["translation_logits"]          # (1, T, vocab)
        pred_ids = torch.argmax(logits, dim=-1)[0]        # (T,)

        # decode
        target_tokens_decoded = [tokenizer.decode([t.item()]) for t in tgt_tokens[0]]
        pred_tokens_decoded   = [tokenizer.decode([t.item()]) for t in pred_ids]
        masked                = ["[MASK]" if l == -100 else tokenizer.decode([t.item()])
                                 for t, l in zip(tgt_tokens[0], tgt_labels[0])]

        print(f"\n{'='*60}")
        print(f"[PROBE] Epoch {epoch+1}")
        print(f"  SRC    : {probe_sample['src_text']}")
        print(f"  TARGET : {probe_target}")
        print(f"  MASKED : {' | '.join(masked)}")
        print(f"  INPUT  : {' | '.join(target_tokens_decoded)}")
        print(f"  PRED   : {' | '.join(pred_tokens_decoded)}")
        
        # per-token match
        matches = [p == t for p, t in zip(pred_tokens_decoded, target_tokens_decoded)]
        accuracy = sum(matches) / len(matches)
        print(f"  MATCH  : {' | '.join('✓' if m else '✗' for m in matches)}")
        print(f"  TOK ACC: {accuracy:.2%}")
        print(f"{'='*60}\n")

    model.train()


print("[INFO] Training started")

for epoch in range(NUM_EPOCHS):
    model.train()

    lam = min(1.0, epoch / 10 * 2)
    print(f"\n[INFO] Epoch {epoch+1} | lambda={lam:.3f}")

    running_loss = 0

    for step, (waveform, texts, spk_labels) in enumerate(train_loader):

        waveform = waveform.to(device)
        spk_labels = spk_labels.to(device)

        tgt_tokens = tokenize(tokenizer, texts).to(device)
        tgt_labels = tgt_tokens.clone()
        tgt_labels[tgt_labels == tokenizer.pad_token_id] = -100

        outputs = model(
            waveform,
            tgt_tokens=tgt_tokens,
            lam=lam
        )

        loss_dict = loss_fn(outputs, spk_labels, tgt_labels)

        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()

        running_loss += loss_dict["total"].item()

        if step % 10 == 0:
            print(f"[STEP {step}] loss={loss_dict['total']:.4f}")

    print(f"[EPOCH DONE] avg_loss={running_loss/len(train_loader):.4f}")
    probe_epoch(model, epoch)   

print("[INFO] Training complete")
torch.save({
    "model": model.state_dict(),
    "spk2id": spk2id,
    "train_samples": small_ds, 
}, "checkpoint_low_epoch.pt")

print("[INFO] Model checkpoint saved to checkpoint.pt")