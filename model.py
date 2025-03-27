import torch.nn as nn
from adapters import Adapter

class TransformerWithAdapters(nn.Module):
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
        output = self.output_layer(tgt.permute(1, 0, 2))
        return output
