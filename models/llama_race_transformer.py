import torch
import torch.nn as nn
import math
import itertools
from race_ext.race_ext import race_pref
import torch.nn.functional as F
from models.shared import LlamaFeedForward, RMSNorm
    
class LlamaRACEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [LlamaRACEBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        ####################################################

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        x = tok_embeds
        x = self.drop_emb(x)

        # NEW
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0

class LlamaRACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att = LlamaRACEAttention(
                       d     = cfg["emb_dim"],
                       h     = cfg["n_heads"],
                       K     = cfg["K"],
                       L     = cfg["L"],
                       M     = cfg["M"],
                       drop  = cfg["drop_rate"],
                       qkv_bias    = cfg["qkv_bias"],
                       device      = device
                   )
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ff    = LlamaFeedForward(cfg)
        self.drop_shortcut  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class LlamaRACEAttention(nn.Module):
    """Multi‑head wrapper around BatchedACE."""
    def __init__(self, d, h, K, L, M, drop=0.1,
                 qkv_bias=False, device='cpu'):
        super().__init__()
        assert d % h == 0
        self.h, self.d_k, self.M = h, d//h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        self.K, self.L= K, L
        self.R = 1 << K

        # calculates rotation 
        # low dimensions rotate smaller and high dimensions rotate more
        theta = 10000 ** (-torch.arange(0, self.d_k, 2, device=device).float() / self.d_k)
        self.register_buffer("theta", theta) 

        # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
        planes = torch.randn(M, L, K, self.d_k, device=device)
        planes = planes.view(M, L*K, self.d_k).transpose(1,2)
        self.register_buffer('planes_T', planes)
        # planes_T: [M, d_k, L*K]

        # flatten protos [R, K] --> [K, R]
        corners = torch.tensor(
            list(itertools.product([-1., +1.], repeat=K)),
            device=device
        )
        self.register_buffer('protos_T', corners.T)  # [K, R]
        self.device = device
        self.eps = 1e-6

        self.A_sum = None
        self.B_sum = None
        self.ptr_current_pos = 0
        

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape
        # split heads
        Q = self.q(x).view(B, T, self.h, self.d_k)
        K = self.k(x).view(B, T, self.h, self.d_k)
        V = self.v(x).view(B, T, self.h, self.d_k)

        # apply positional encoding
        start = self.ptr_current_pos
        pos = torch.arange(start, start + T, device=x.device).unsqueeze(1)
        freq = pos * self.theta.unsqueeze(0)
        sin = freq.sin()[None, :, None, :]
        cos = freq.cos()[None, :, None, :]

        Q_even, Q_odd = Q[..., ::2], Q[..., 1::2]
        K_even, K_odd = K[..., ::2], K[..., 1::2]

        Q = torch.cat([Q_even*cos - Q_odd*sin, Q_even*sin + Q_odd*cos], dim=-1)
        K = torch.cat([K_even*cos - K_odd*sin, K_even*sin + K_odd*cos], dim=-1)


        # pack M ensembles
        pack = lambda z: z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)
        Khf, Vhf, Qhf = pack(K), pack(V), pack(Q)
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        BH = B * H
        N  = M * BH
        S  = self.L * self.R
        scale = math.sqrt(dk)

        # ---- Flatten (only across B*H), keep time T explicit ----
        # [M,B,T,H,d] → [M,BH,T,d] → [N,T,d]
        Kh = Khf.permute(0,1,3,2,4).reshape(M, BH, T, dk).reshape(N, T, dk)
        V  = Vhf.permute(0,1,3,2,4).reshape(M, BH, T, dk).reshape(N, T, dk)
        Qh = Qhf.permute(0,1,3,2,4).reshape(M, BH, T, dk).reshape(N, T, dk)

        # ---- One batched GEMM per ensemble for K and Q projections ----
        # Reshape back to [M,BH,T,d] to hit per-ensemble planes
        Kh4 = Kh.view(M, BH, T, dk)   # [M,BH,T,d]
        Qh4 = Qh.view(M, BH, T, dk)   # [M,BH,T,d]
        projK = torch.einsum('mbtd, mds -> mbts', Kh4, self.planes_T)  # [M,BH,T,L*K]
        projQ = torch.einsum('mbtd, mds -> mbts', Qh4, self.planes_T)  # [M,BH,T,L*K]

        # → [N,T,L,K]
        projK = projK.reshape(N, T, self.L, self.K)
        projQ = projQ.reshape(N, T, self.L, self.K)

        # ---- Soft hash to R prototypes per plane ----
        probsK = F.softmax((projK.tanh() / scale) @ self.protos_T, dim=-1)  # [N,T,L,R]
        probsQ = F.softmax((projQ.tanh() / scale) @ self.protos_T, dim=-1)  # [N,T,L,R]

        if not use_cache:
            # A_pref(t) = sum_{<=t} probsK
            A_pref = probsK.cumsum(dim=1)                                               # [N,T,L,R]
            # B_pref(t) = sum_{<=t} probsK * V
            B_pref = (probsK.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)).cumsum(dim=1) # [N,T,L,R,d]
            # E_pref(t) = B_pref / (A_pref + eps)
            E_pref = B_pref / (A_pref.unsqueeze(-1) + self.eps)                         # [N,T,L,R,d]
            # out(t) = probsQ(t) @ E_pref(t)  over S=L*R
            out2 = torch.bmm(
                probsQ.view(N*T, 1, S),
                E_pref.contiguous().view(N*T, S, dk)
            ).view(N, T, dk)                                                            # [N,T,d]

            # Unflatten and merge heads+ensembles, project, dropout (match cached path)
            out_h = out2.view(M, BH, T, dk).view(M, B, H, T, dk).permute(0,1,3,2,4)  # [M,B,T,H,d]
            out   = out_h.mean(0).permute(0,2,1,3).contiguous().view(B, T, H*dk)     # [B,T,d_model]
            return self.drop(self.o(out))

        # Initialize/resize cache if needed (works across calls)
        need_new = (
            self.A_sum is None or
            self.A_sum.shape[0] != N
        )
        if need_new:
            self.A_sum = torch.zeros(N, self.L, self.R, device=self.device)
            self.B_sum = torch.zeros(N, self.L, self.R, dk, device=self.device)

        if self.ptr_current_pos == 0:
            out2 = race_pref.race_causal_fused_forward(
                probsK.contiguous().view(N, T, S), probsQ.contiguous().view(N, T, S), V.contiguous(), 1e-6
            )
            self.A_sum += probsK.sum(dim=1)
            self.B_sum += (probsK.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3)).sum(dim=1) 
            self.ptr_current_pos += T
        else:
            assert T == 1, "DECODE"
            # Last-step soft bucket weights and weighted values
            A_last = probsK[:, -1]  # [N, L, R]
            B_last = (probsK.unsqueeze(-1) * V.unsqueeze(2).unsqueeze(3))[:, -1]  # [N, L, R, dk]

            # Use the cache (prefix totals) + new token to get updated prefix stats
            A_new = self.A_sum + A_last                              # [N, L, R]
            B_new = self.B_sum + B_last                               # [N, L, R, dk]

            # Prefix mean for this step (this is what the query "reads")
            E_step = B_new / (A_new.unsqueeze(-1) + self.eps)         # [N, L, R, dk]

            # Output for the new token: probsQ at last step @ E_step over S=L*R
            out2 = torch.bmm(
                probsQ.reshape(N, 1, S),                               # [N,1,S]
                E_step.reshape(N, S, dk)                               # [N,S,dk]
            ).reshape(N, 1, dk)                                        # [N,1,dk]

            # Update cache for the next token
            self.A_sum = A_new
            self.B_sum = B_new
            self.ptr_current_pos += 1

        # Unflatten
        out_h = out2.view(M, BH, T, dk).view(M, B, H, T, dk).permute(0,1,3,2,4)  
        # merge ensembles & heads
        out   = out_h.mean(0).permute(0,2,1,3).contiguous().view(B, T, -1)
        return self.drop(self.o(out))

    def reset_cache(self):
        self.A_sum = None
        self.B_sum = None
        self.ptr_current_pos = 0  
