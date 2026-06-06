import torch

class ExpressoDataset(torch.utils.data.Dataset):
    def __init__(self, data, spk2id):
        self.data = data
        self.spk2id = spk2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        waveform = torch.tensor(x["audio"]["array"], dtype=torch.float32)
        text = x["src_text"]
        spk = torch.tensor(self.spk2id[x["src_speaker"]], dtype=torch.long)

        return waveform, text, spk


