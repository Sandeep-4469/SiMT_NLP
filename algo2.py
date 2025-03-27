import torch
import sentencepiece as spm

def adaptive_inference(model, src_tokens, sp, kmin, kmax, rho_k, return_positions=False):
    """
    Algorithm 2: Uncertainty-Based Adaptive Policy
    
    Args:
        model: Trained Transformer model with adapters
        src_tokens: Tokenized source sentence (list of token IDs)
        sp: SentencePiece processor for decoding
        kmin: Minimum read steps before writing
        kmax: Maximum read steps allowed
        rho_k: Uncertainty threshold for read/write decision
        return_positions: If True, returns read positions along with output tokens
    
    Returns:
        y: Generated target sequence (list of token IDs)
        read_positions (if return_positions=True): List of positions where reading happened
    """
    model.eval()
    y = []
    k = 1
    READ_TOKEN_ID = sp.piece_to_id("<pad>")  # Using PAD token for READ action
    EOS_TOKEN_ID = sp.piece_to_id("</s>")  # End-of-sequence token ID
    read_positions = []
    
    print(f"🔍 Initial src_tokens: {src_tokens}")
    
    while len(src_tokens) < kmax and (not y or y[-1] != EOS_TOKEN_ID):
        if k < kmin:
            src_tokens.append(READ_TOKEN_ID)
        else:
            src_tensor = torch.tensor(src_tokens, device=model.device).unsqueeze(0)
            y_tensor = torch.tensor(y, device=model.device).unsqueeze(0) if y else None
            
            with torch.no_grad():
                output_probs = model(src_tensor, y_tensor, None, None, None, k)
            
            ytop = torch.argmax(output_probs, dim=-1).cpu().item()  # Ensure tensor is on CPU
            ptop = torch.max(output_probs).item()
            
            if k < kmax and ptop < rho_k[k]:
                src_tokens.append(READ_TOKEN_ID)
            else:
                y.append(ytop)
                
        read_positions.append(len(src_tokens))
        k = len(src_tokens) - len(y)
        print(f"Step {k}: src_tokens={src_tokens}, y={y}, read_positions={read_positions}")
    
    return (y, read_positions) if return_positions else y

# Load SentencePiece model to retrieve EOS_TOKEN_ID
sp = spm.SentencePieceProcessor(model_file="bpe.model")
EOS_TOKEN_ID = sp.piece_to_id("</s>")
print("EOS Token ID:", EOS_TOKEN_ID)
