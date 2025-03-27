import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from algo2 import adaptive_inference
from model import TransformerWithAdapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sentencepiece_model():
    sp = spm.SentencePieceProcessor(model_file="bpe.model")
    return sp, sp.piece_to_id("</s>")

sp, EOS_TOKEN_ID = load_sentencepiece_model()

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0).to(device)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0).to(device)
    return src_batch, tgt_batch

def load_wmt15_dataset():
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

def train_bpe_model():
    if os.path.exists("bpe.model"):
        return
    os.system("spm_train --input=data/train.de,data/train.en --model_prefix=bpe --vocab_size=32000 --character_coverage=1.0")

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

load_wmt15_dataset()
train_bpe_model()
sp, EOS_TOKEN_ID = load_sentencepiece_model()

train_dataset = WMT15Dataset("train", sp)
val_dataset = WMT15Dataset("validation", sp)
test_dataset = WMT15Dataset("test", sp)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = TransformerWithAdapters(d_model=512, num_heads=8, d_ff=2048, num_layers=4, dropout=0.1, num_adapters=4).to(device)

model_path = "best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

def train_model(model, dataloader, optimizer, criterion, num_epochs=5):
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

train_model(model, train_dataloader, optimizer, criterion)

torch.save(model.state_dict(), model_path)
print("Model training complete and saved")

sample_src_sentence = test_dataset.src_sentences[13]
src_tokens = sp.encode(sample_src_sentence)

adaptive_result, read_positions = adaptive_inference(
    model, src_tokens, sp, kmin=3, kmax=10, rho_k=[0.9] * 10, return_positions=True
)

decoded_output = sp.decode(adaptive_result)

print("Generated Translation:", decoded_output)
print("Read Positions:", read_positions)
