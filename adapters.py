import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_size=64):
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, d_model)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x))) + x
