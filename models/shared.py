import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    
# For llama
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # 3 linear projections, 2 expansion 1 compression
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim) 

    def forward(self, x):
        a = self.w1(x)
        b = torch.sigmoid(self.w2(x))
        return self.w3(a * b)
    
class LlamaFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["emb_dim"] * 4)
        self.ff = SwiGLU(cfg["emb_dim"], hidden)

    def forward(self, x):
        return self.ff(x)


# need to replace layerNorm with RMSNorm for llama
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.eps = eps

    def forward(self, x):
        # array of rms for each vector
        rms = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return x / (rms + self.eps) * self.scale