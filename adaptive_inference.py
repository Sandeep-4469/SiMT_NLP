import torch
from model import TransformerWithAdapters
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerWithAdapters(d_model=512, num_heads=8, d_ff=2048, num_layers=4, dropout=0.1, num_adapters=4).to(device)
model.load_state_dict(torch.load("trained_model.pth"))

sp = spm.SentencePieceProcessor(model_file="bpe.model")

def adaptive_inference(src_tokens, kmin=3, kmax=10, rho_k=[0.9] * 10):
    y = []
    k = 1
    while k < kmax and (not y or y[-1] != sp.piece_to_id("</s>")):
        with torch.no_grad():
            output_probs = model(torch.tensor(src_tokens).unsqueeze(0).to(device), None, None, None, None, k)
        y.append(torch.argmax(output_probs, dim=-1).cpu().item())
        k += 1
    return sp.decode(y)

print(adaptive_inference(sp.encode("Das ist ein Test.")))
