import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import os
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from algo2 import adaptive_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    """Pads sequences to the same length within a batch."""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0).to(device)
    return src_batch, tgt_batch

def load_wmt15_from_huggingface():
    """Downloads and saves the WMT15 De-En dataset if not already present."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/train.de") and os.path.exists("data/train.en"):
        return
    dataset = load_dataset("wmt15", "de-en")
    splits = {"train": 20000, "validation": 1000, "test": 1000}
    for split, size in splits.items():
        with open(f"data/{split}.de", "w") as f_de, open(f"data/{split}.en", "w") as f_en:
            for example in dataset[split].select(range(size)):
                f_de.write(example["translation"]["de"] + "\n")
                f_en.write(example["translation"]["en"] + "\n")

def train_bpe():
    """Trains a BPE model on the dataset if not already trained."""
    if os.path.exists("bpe.model"):
        return
    os.system("spm_train --input=data/train.de,data/train.en --model_prefix=bpe --vocab_size=32000 --character_coverage=1.0")

class WMT15Dataset(Dataset):
    """Dataset class to process WMT15 De-En dataset with BPE tokenization."""
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

class Adapter(nn.Module):
    """Adapter module for modifying transformer outputs."""
    def __init__(self, d_model, bottleneck_size=64):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x))) + x

class TransformerWithAdapters(nn.Module):
    """Transformer model with adapters for simultaneous translation."""
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, num_adapters, bottleneck_size=64):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout), num_layers
        )
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.adapters = nn.ModuleList([Adapter(d_model, bottleneck_size) for _ in range(num_adapters)])
        self.embedding = nn.Embedding(32000, d_model)
        self.output_layer = nn.Linear(d_model, 32000)
    
    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, wait_k_idx):
        src = self.embedding(src).permute(1, 0, 2)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        memory = self.encoder(src)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask)
        tgt = self.adapters[wait_k_idx](tgt)
        return self.output_layer(tgt.permute(1, 0, 2))

def train_adapters_wait_k(model, dataloader, optimizer, criterion, num_epochs=5):
    """Trains the transformer model with adapters using wait-k strategy."""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt, None, None, None, 0)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

load_wmt15_from_huggingface()
train_bpe()
sp = spm.SentencePieceProcessor(model_file="bpe.model")
train_dataset = WMT15Dataset("train", sp)
val_dataset = WMT15Dataset("validation", sp)
test_dataset = WMT15Dataset("test", sp)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = TransformerWithAdapters(d_model=512, num_heads=8, d_ff=2048, num_layers=4, dropout=0.1, num_adapters=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

torch.cuda.empty_cache()
torch.cuda.memory_reserved(device=None)

train_adapters_wait_k(model, train_dataloader, optimizer, criterion)

sample_src_sentence = test_dataset.src_sentences[13]
src_tokens = sp.encode(sample_src_sentence)
adaptive_result, read_positions = adaptive_inference(model, src_tokens, sp, kmin=3, kmax=10, rho_k=[0.9] * 10, return_positions=True)

decoded_output = sp.decode(adaptive_result)

print("Generated Translation:", decoded_output)
print("Read Positions:", read_positions)
