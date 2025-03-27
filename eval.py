import torch
import sacrebleu
from torch.utils.data import DataLoader
from algo2 import adaptive_inference
from train2 import model, test_dataloader, sp

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to compute BLEU score
def compute_bleu(hypotheses, references):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

# Function to evaluate the model
def evaluate_model(model, dataloader, sp, kmin=3, kmax=10, rho_k=[0.9]*10):
    model.eval()
    references = []
    hypotheses = []
    
    for src_tokens, tgt_tokens in dataloader:
        src_tokens = src_tokens.to(device)  # Use device instead of model.device
        tgt_tokens = tgt_tokens.to(device)
        
        # Decode target sentence
        tgt_text = [sp.decode(t.tolist()) for t in tgt_tokens]
        references.extend(tgt_text)  # SacreBLEU expects list format
        
        # Perform adaptive inference
        pred_tokens, _ = adaptive_inference(model, src_tokens.squeeze(0).tolist(), sp, kmin, kmax, rho_k, return_positions=False)
        pred_text = sp.decode(pred_tokens)
        hypotheses.append(pred_text)
    
    # Compute BLEU Score
    bleu_score = compute_bleu(hypotheses, references)
    print(f"BLEU Score: {bleu_score:.2f}")
    
    return {"BLEU": bleu_score}

if __name__ == "__main__":
    print("Evaluating Model...")
    metrics = evaluate_model(model, test_dataloader, sp)
    print("Evaluation Completed!")
    print(metrics)
