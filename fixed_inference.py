import torch
from model import TransformerWithAdapters
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerWithAdapters(d_model=512, num_heads=8, d_ff=2048, num_layers=4, dropout=0.1, num_adapters=4).to(device)
model.load_state_dict(torch.load("trained_model.pth"))

sp = spm.SentencePieceProcessor(model_file="bpe.model")

def translate(sentence):
    src_tokens = torch.tensor(sp.encode(sentence)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(src_tokens, None, None, None, None, 0)
    return sp.decode(torch.argmax(output, dim=-1).tolist()[0])

print(translate("Das ist ein Test."))
