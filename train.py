import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from dataset import WMT15Dataset
from model import TransformerWithAdapters
from utils import collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor(model_file="bpe.model")
train_dataset = WMT15Dataset("train", sp)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = TransformerWithAdapters(d_model=512, num_heads=8, d_ff=2048, num_layers=4, dropout=0.1, num_adapters=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

def train(model, dataloader, optimizer, criterion, num_epochs=5):
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

train(model, train_dataloader, optimizer, criterion)
torch.save(model.state_dict(), "trained_model.pth")
