import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class MACHAttention(nn.Module):
    def __init__(self, d_k, heads, K_bits, L, dropout, qkv_bias=False, top_k=2, device='cuda'):
        super().__init__()
        self.d_k = d_k              # model dim
        self.num_heads = heads
        self.head_dim = d_k // heads

        self.K_bits = K_bits
        self.L = L
        self.R = 1 << K_bits
        self.S = self.L * self.R
        self.top_k = top_k          # per-(token,hash) buckets to keep; None = no sparsification

        # Random planes: [L, K_bits, head_dim] -> [head_dim, L*K_bits]
        planes = torch.randn(L, K_bits, self.head_dim, device=device)
        self.register_buffer(
            'planes_T',
            planes.view(L * K_bits, self.head_dim).T  # [head_dim, L*K_bits]
        )

        # Prototypes (corners of {-1,+1}^K_bits): [K_bits, R]
        corners = torch.tensor(
            list(itertools.product([-1.0, +1.0], repeat=K_bits)),
            device=device
        )  # [R, K_bits]
        self.register_buffer('protos_T', corners.T)  # [K_bits, R]

        self.q_proj = nn.Linear(d_k, d_k, bias=qkv_bias)
        self.k_proj = nn.Linear(d_k, d_k, bias=qkv_bias)
        self.v_proj = nn.Linear(d_k, d_k, bias=qkv_bias)
        self.out    = nn.Linear(d_k, d_k)
        self.drop   = nn.Dropout(dropout)

        # Learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        dim = self.head_dim

        # 1) Q, K, V: [B, T, D] -> [B, H, T, dim]
        Q = self.q_proj(x).view(B, T, H, dim).transpose(1, 2)  # [B, H, T, dim]
        K = self.k_proj(x).view(B, T, H, dim).transpose(1, 2)  # [B, H, T, dim]
        V = self.v_proj(x).view(B, T, H, dim).transpose(1, 2)  # [B, H, T, dim]

        scale = self.logit_temp.exp().clamp(1e-2, 20.0)
        S, L, K_bits, R = self.S, self.L, self.K_bits, self.R

        # 2) Flatten batch & heads: [B, H, T, dim] -> [N, T, dim]
        N = B * H
        K_flat = K.contiguous().view(N, T, dim)   # [N, T, dim]

        # 3) Project keys using planes: [N, T, dim] @ [dim, L*K_bits] -> [N, T, L*K_bits]
        projK = K_flat @ self.planes_T            # [N, T, L*K_bits]

        # 4) Reshape to [N, T, L, K_bits]
        projK = projK.view(N, T, L, K_bits)       # [N, T, L, K_bits]

        # 5) Soft hashing via hypercube prototypes
        logitsK = (projK.tanh() / scale) @ self.protos_T   # [N, T, L, R]
        probsK  = F.softmax(logitsK, dim=-1)               # [N, T, L, R]

        # 5.5) Top-k sparsification over R per (N,T,L)
        if (self.top_k is not None) and (0 < self.top_k < R):
            # indices of top_k buckets along R
            _, top_idx = probsK.topk(self.top_k, dim=-1)   # [N, T, L, top_k]

            # mask 1 at top_k positions, 0 elsewhere
            mask = torch.zeros_like(probsK)
            mask.scatter_(-1, top_idx, 1.0)

            # keep only top_k entries
            probsK = probsK * mask

            # renormalize over R so it still sums to 1
            probsK = probsK / (probsK.sum(dim=-1, keepdim=True) + 1e-6)

        # 6) Collapse (L, R) -> S buckets
        probsK_S = probsK.view(N, T, S)                    # [N, T, S]
        V_flat = V.contiguous().view(N, T, dim)             # [BH, T, dim]
        Q_flat = Q.contiguous().view(N, T, dim)             # [BH, T, dim]

        # 7) Build bucketed K and V by weighted sums over time
        bucket_K = probsK_S.transpose(1, 2).bmm(K_flat)      # [BH, S, dim]
        bucket_V = probsK_S.transpose(1, 2).bmm(V_flat)      # [BH, S, dim]

        # Normalize buckets: average instead of sum
        A = probsK_S.sum(dim=1)                              # [BH, S]
        bucket_K = bucket_K / (A.unsqueeze(-1) + 1e-6)       # [BH, S, dim]
        bucket_V = bucket_V / (A.unsqueeze(-1) + 1e-6)       # [BH, S, dim]

        # 8) Query interacts with bucketed keys
        # scores: [BH, T, S] = [BH, T, dim] @ [BH, dim, S]
        scores = Q_flat.bmm(bucket_K.transpose(1, 2))        # [BH, T, S]
        scores = scores / math.sqrt(dim)
        probs_attn = F.softmax(scores, dim=-1)               # [BH, T, S]

        # 9) Mix bucketed values
        out_flat = probs_attn.bmm(bucket_V)                  # [BH, T, dim]

        # 10) Merge heads: [B, H, T, dim] -> [B, T, D]
        out = out_flat.view(B, H, T, dim).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.drop(out)
