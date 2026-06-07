# CCSRD: Content and Speaker Representation Disentanglement

Speech-to-text translation (English audio → German text) that separates linguistic content from speaker identity using adversarial training.

## Quick Start

```bash
# Install
pip install torchaudio>=2.11.0 transformers>=5.9.0 datasets

# Train
python train.py

# Inference
python infer.py
```

## Model Architecture

```
Waveform (audio)
    ↓
SpeechEncoder (Wav2Vec2 + Conv1d)  →  H (speech features)
    ↓
┌───────────────────┐
│ ContentEncoder    │ (Transformer, 6 layers) → Z_c (what is said)
│ NonContentEncoder │ (Transformer, 6 layers) → Z_nc (who said it)
└───────────────────┘
    ↓         ↓
    │         └──→ SpeakerClassifier → speaker prediction (L_SPK)
    │
    ├──→ CyclicReconstructor (with GRL)
    │    ├─ Predict Z_c from Z_nc → L_CON
    │    └─ Predict Z_nc from Z_c → L_NCON
    │
    ├──→ FeatureReconstructor
    │    └─ Reconstruct H from Z_c ⊕ Z_nc → L_REC
    │
    └──→ TranslationDecoder (6 layers, causal mask)
         └─ German translation logits → L_ST
```

## Components Explained

| File | Purpose |
|------|---------|
| **models/speech_encoder.py** | Wav2Vec2 (frozen) + Conv layers to get speech features H |
| **models/encoders.py** | ContentEncoder (linguistic), NonContentEncoder (speaker), TranslationDecoder (translation) |
| **models/reconstructors.py** | CyclicReconstructor (adversarial GRL to separate Z_c/Z_nc), FeatureReconstructor (reconstruct H) |
| **models/classifier.py** | SpeakerClassifier: pool Z_nc → predict speaker ID |
| **models/ccsrd.py** | Main model combining all components |
| **models/utils.py** | CCSRDLoss: combines L_ST + L_CON + L_NCON + L_REC + L_SPK |

| File | Purpose |
|------|---------|
| **data/dataset.py** | ExpressoDataset: returns (waveform, text, speaker_id) |
| **data/utils.py** | load_data_samples() (from mExpresso), build_speaker_map(), collate_fn for DataLoader |

| File | Purpose |
|------|---------|
| **train.py** | Load data → create model → training loop (20 epochs, batch_size=2) |
| **infer.py** | Load checkpoint → teacher forcing check → autoregressive generation |
| **utils.py** | tokenize() helper, infer() function for generation |

## How to Use the Code

### Training

`train.py` does:
1. Load ~3000 samples from mExpresso (English-German pairs)
2. Map speakers to IDs (e.g., speaker_1 → 0, speaker_2 → 1)
3. Create DataLoader with batches of 2
4. Initialize CCSRD model (d_model=512, vocab_size=tokenizer.vocab_size)
5. Training loop:
   - Forward pass: `outputs = model(waveform, tgt_tokens=tokens, lam=1.0)`
   - Compute loss via CCSRDLoss
   - Backward + optimizer step
   - Probe sample check every epoch (teacher forcing on a fixed sample)
6. Save checkpoint as `checkpoint.pt`

Key hyperparameters:
- `NUM_EPOCHS = 20`
- `batch_size = 2`
- `learning_rate = 1e-4`
- `d_model = 512`

### Inference

`infer.py` does:
1. Load checkpoint (has `spk2id` and `train_samples`)
2. Create model with correct num_speakers and vocab_size
3. Load state dict from checkpoint
4. **Teacher forcing check**: Run model with ground truth tokens → measure accuracy
5. **Autoregressive generation**: Use `infer()` function to generate German text token-by-token without ground truth

The `infer()` function in `utils.py`:
- Takes waveform, tokenizer, device
- Encodes: `H = speech_encoder(waveform)`, `Z_c = content_encoder(H)`
- Generates tokens autoregressively using `translation_decoder`
- Stops at EOS token or max_len=50

### Data

mExpresso dataset format:
```python
{
  "audio": {"array": [...], "sampling_rate": 16000},
  "src_text": "I love coffee",
  "tgt_text": {"deu": "Ich liebe Kaffee"},
  "src_speaker": "john_doe"
}
```

Processing:
- `load_data_samples(num_samples)` → list of dicts
- `build_speaker_map(samples)` → `{speaker_name: int_id}`
- `ExpressoDataset(samples, spk2id)` → yields (waveform_tensor, german_text_str, speaker_id_tensor)
- `collate(batch)` → pads waveforms, stacks speakers

## Loss Function

```
L_total = δ·L_ST + α·(L_CON + L_NCON + L_REC + L_SPK)

L_ST   = CrossEntropy(translation_logits, target_tokens)
L_CON  = MSE(Z_c_pred, Z_c)           [from CyclicReconstructor, via GRL]
L_NCON = MSE(Z_nc_pred, Z_nc)         [from CyclicReconstructor, via GRL]
L_REC  = MSE(H_reconstructed, H)      [from FeatureReconstructor]
L_SPK  = NLLLoss(speaker_logits, speaker_label)  [from SpeakerClassifier]
```

GRL (Gradient Reversal Layer) in CyclicReconstructor:
- Forward pass: identity
- Backward pass: negate gradients (scaled by λ)
- Effect: encoders learn to make Z_c and Z_nc independently non-reconstructible from each other

## File Structure

```
ccsrd/
├── train.py              # Run training
├── infer.py              # Run inference
├── utils.py              # tokenize(), infer() helper functions
├── main.py               # (placeholder, not used)
├── test.py               # (tests)
├── pyproject.toml        # Dependencies
├── checkpoint.pt         # Latest checkpoint
├── checkpoint_low_epoch.pt # Best checkpoint
│
├── models/
│   ├── ccsrd.py              # CCSRD (main model class)
│   ├── speech_encoder.py      # SpeechEncoder
│   ├── encoders.py           # ContentEncoder, NonContentEncoder, TranslationDecoder
│   ├── reconstructors.py      # GRL, CyclicReconstructor, FeatureReconstructor
│   ├── classifier.py          # SpeakerClassifier
│   ├── utils.py              # CCSRDLoss
│   ├── __init__.py           # Exports all classes
│   └── test.py               # (tests)
│
└── data/
    ├── dataset.py            # ExpressoDataset
    ├── utils.py              # load_data_samples, build_speaker_map, collate
    ├── __init__.py           # Exports dataset classes
    └── test.py               # (tests)
```

## Model Forward Pass Example

```python
from models import CCSRD, CCSRDLoss
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = CCSRD(d_model=512, num_speakers=100, vocab_size=tokenizer.vocab_size)
loss_fn = CCSRDLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

# Training
waveform = torch.randn(batch_size, audio_length)
tgt_tokens = tokenizer([...], return_tensors="pt", padding=True)["input_ids"]

outputs = model(waveform, tgt_tokens=tgt_tokens, lam=1.0)
# outputs keys: H, Z_c, Z_nc, z_c_pred, z_nc_pred, H_recon, speaker_logits, translation_logits

loss_dict = loss_fn(outputs, speaker_labels, tgt_tokens)
# loss_dict keys: total, L_CON, L_NCON, L_REC, L_SPK, L_ST

# Inference (content encoding)
Z_c = model.encode_audio(waveform)

# Inference (translation with tokens)
logits = model.translate(waveform, tgt_tokens)
```

## Dependencies

From `pyproject.toml`:
- `torchaudio>=2.11.0`
- `transformers>=5.9.0`
- Pre-trained models: `facebook/wav2vec2-base`, `Helsinki-NLP/opus-mt-en-de`
- Dataset: `jorirsan/mExpresso` from HuggingFace

---

**Summary**: CCSRD learns to separate what is said (Z_c) from who said it (Z_nc) through multi-task learning with adversarial cyclic reconstruction, then uses Z_c for speech-to-text translation.
