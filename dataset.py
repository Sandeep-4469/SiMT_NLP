import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class WMT15Dataset(Dataset):
    def __init__(self, split, sp_model):
        dataset = load_dataset("wmt15", "de-en")[split].select(range(20000 if split == "train" else 1000))
        self.src_sentences = [ex["translation"]["de"] for ex in dataset]
        self.tgt_sentences = [ex["translation"]["en"] for ex in dataset]
        self.sp = sp_model
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_tokens = self.sp.encode(self.src_sentences[idx].strip())
        tgt_tokens = self.sp.encode(self.tgt_sentences[idx].strip())
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)
