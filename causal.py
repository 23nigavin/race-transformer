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
        """
        Causal MACH attention (prefix buckets):
        - Each time step t only sees buckets built from tokens <= t.
        - Uses N = B*H flattening and the same top_k mechanism as the non-causal code.

        Args:
            x:    [B, T, D]
            mask: [B, T] (1 = keep, 0 = pad) or None
        """
        B, T, D = x.shape
        H = self.num_heads
        dim = self.head_dim
        N = B * H
        S, L, K_bits, R = self.S, self.L, self.K_bits, self.R

        # 1) Q, K, V: [B, T, D] -> [B, H, T, dim]
        Q = self.q_proj(x).view(B, T, H, dim)
        K = self.k_proj(x).view(B, T, H, dim)
        V = self.v_proj(x).view(B, T, H, dim)

        Q = Q.transpose(1, 2)  # [B, H, T, dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 2) Flatten batch & heads: [B, H, T, dim] -> [N, T, dim]
        Q_flat = Q.contiguous().view(N, T, dim)   # [N, T, dim]
        K_flat = K.contiguous().view(N, T, dim)   # [N, T, dim]
        V_flat = V.contiguous().view(N, T, dim)   # [N, T, dim]

        # 3) Project keys using planes: [N, T, dim] @ [dim, L*K_bits] -> [N, T, L*K_bits]
        projK = K_flat @ self.planes_T            # [N, T, L*K_bits]

        # 4) Reshape to [N, T, L, K_bits]
        projK = projK.view(N, T, L, K_bits)       # [N, T, L, K_bits]

        # 5) Soft hashing via hypercube prototypes
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)
        logitsK = (projK.tanh() / scale) @ self.protos_T   # [N, T, L, R]
        probsK  = F.softmax(logitsK, dim=-1)               # [N, T, L, R]

        # 5.5) Top-k sparsification over R per (N,T,L)  (same mechanism as non-causal)
        if (self.top_k is not None) and (0 < self.top_k < R):
            # indices of top_k buckets along R
            _, top_idx = probsK.topk(self.top_k, dim=-1)   # [N, T, L, top_k]

            # mask 1 at top_k positions, 0 elsewhere
            mask_top = torch.zeros_like(probsK)
            mask_top.scatter_(-1, top_idx, 1.0)

            # keep only top_k entries
            probsK = probsK * mask_top

            # renormalize over R so it still sums to 1
            probsK = probsK / (probsK.sum(dim=-1, keepdim=True) + 1e-6)

        # 6) Collapse (L, R) -> S buckets
        probsK_S = probsK.view(N, T, S)                    # [N, T, S]
        assign = probsK_S                                  # [N, T, S]

        # 7) Build *causal* bucketed K and V via prefix sums in time
        #    bucket_K[n,t,s] = sum_{τ<=t} assign[n,τ,s] * K_flat[n,τ,:] / sum_{τ<=t} assign[n,τ,s]

        # Expand dims for broadcasting
        assign_exp = assign.unsqueeze(-1)                  # [N, T, S, 1]
        K_exp      = K_flat.unsqueeze(2)                   # [N, T, 1, dim]
        V_exp      = V_flat.unsqueeze(2)                   # [N, T, 1, dim]

        # Weighted contributions per time step
        weighted_K = assign_exp * K_exp                    # [N, T, S, dim]
        weighted_V = assign_exp * V_exp                    # [N, T, S, dim]

        # Prefix sums along time dimension (causal buckets)
        num_K = weighted_K.cumsum(dim=1)                   # [N, T, S, dim]
        num_V = weighted_V.cumsum(dim=1)                   # [N, T, S, dim]

        denom = assign.cumsum(dim=1)                       # [N, T, S]
        denom = denom.unsqueeze(-1)                        # [N, T, S, 1]

        bucket_K = num_K / (denom + 1e-6)                  # [N, T, S, dim]
        bucket_V = num_V / (denom + 1e-6)                  # [N, T, S, dim]

        # 8) Query interacts with *per-time* buckets
        # scores[n,t,s] = <Q_flat[n,t,:], bucket_K[n,t,s,:]> / sqrt(dim)
        scores = torch.einsum("ntd,ntsd->nts", Q_flat, bucket_K)  # [N, T, S]
        scores = scores / math.sqrt(dim)
        probs_attn = F.softmax(scores, dim=-1)             # [N, T, S]

        # 9) Mix bucketed values per time step
        out_flat = torch.einsum("nts,ntsd->ntd", probs_attn, bucket_V)  # [N, T, dim]

        # 10) Merge heads: [N, T, dim] -> [B, T, D]
        out = out_flat.view(B, H, T, dim).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.drop(out)
