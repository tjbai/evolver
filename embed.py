import math

import torch
import torch.nn as nn

class ReversibleEmbedding(nn.Module):
    
    def forward(self, x, d):
        raise NotImplementedError()

class SinusoidalEmbedding(ReversibleEmbedding):
    
    def __init__(self, d_model=512, max_len=10):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    
    def forward(self, x, d):
        return x + d * self.pe[:, :x.shape[-2], :]
    
class RotaryEmbedding(ReversibleEmbedding):
    
    def __init__(self):
        pass
    
    def forward(self, x, d):
        pass
    
class IdentityEmbedding(ReversibleEmbedding):
    def __init__(self, **_): super().__init__()
    def forward(self, x, **_): return x
    
class LearnedEmbedding(ReversibleEmbedding):
    
    def __init__(self, d_model=512, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000) / d_model))
        self.embedding.weight.data[:, 0::2] = torch.sin(pos * div)
        self.embedding.weight.data[:, 1::2] = torch.cos(pos * div)
        
    def forward(self, x, d, pos=None):
        if pos is None:
            B, N, _ = x.shape
            pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + d * self.embedding(pos)
   
class DepthEmbedding(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model, bias=True), nn.ReLU(), nn.Linear(d_model, d_model, bias=True))
   
    # a bit different than standard?
    # whatever
    def sinusoidal_embedding(self, t, max_period=10_000):
        half = self.d_model // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, t):
        t = self.sinusoidal_embedding(t)
        return self.ffn(t)
