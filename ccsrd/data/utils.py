import torch
import torch.nn.utils.rnn as rnn_utils
from datasets import load_dataset

def load_data_samples(num_samples=300):
    ds = load_dataset(
        "jorirsan/mExpresso",
        split="validation",
        streaming=True
    )
    return list(ds.take(num_samples))
    
def build_speaker_map(dataset):
    print("[INFO] Building speaker map...")

    speakers = sorted(set(x["src_speaker"] for x in dataset))
    spk2id = {s: i for i, s in enumerate(speakers)}
    num_speakers = len(spk2id)

    print(f"[INFO] Speakers: {num_speakers}")

    return spk2id, num_speakers


def collate(batch):
    wave, text, spk = zip(*batch)

    wave = rnn_utils.pad_sequence(wave, batch_first=True)
    spk = torch.stack(spk)

    return wave, text, spk