# ==================================================
# 0) Imports & Global Config
# ==================================================
import math, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention import sdpa_kernel, SDPBackend

from .arxiv_config import DEVICE

# ==================================================
# 5) Attention modules (all baselines from vision)
#     – text version is pad-mask aware
# ==================================================

class MultiHeadAttention(nn.Module):
    """Standard softmax MH attention with pad mask, using SDPA."""
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        """
        x:    [B, T, d]
        mask: [B, T] with 1 for real tokens, 0 for PAD
        """
        B, T, _ = x.shape
        h, dk = self.h, self.dk

        # [B, T, d] -> [B, H, T, D]
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        Q, K, V = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,      # we keep dropout on the output like before
                is_causal=False,
            )
        # [B, H, T, D] -> [B, T, H*D]
        out = out.transpose(1, 2).contiguous().view(B, T, h * dk)
        out = self.drop(out)
        out = out.to(self.o.weight.dtype)
        return self.o(out)

class AngularAttention(nn.Module):
    """Angular (cosine) attention."""
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        assert d % h == 0
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B, T, _ = x.shape
        h, dk = self.h, self.dk

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2)
        K = self.k(x).view(B, T, h, dk).transpose(1, 2)
        V = self.v(x).view(B, T, h, dk).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        sim = (Q @ K.transpose(-2, -1)).clamp(-0.999, 0.999)
        scores = 1.0 - torch.acos(sim) / math.pi
        if mask is not None:
            pad = mask[:, None, None, :]
            scores = scores.masked_fill(pad == 0, 0.0)

        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1, keepdim=True) + 1e-6)
        W = self.drop(W)

        out = (W @ V).transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.o(out)

class BatchedACE(nn.Module):
    """Non-causal ACE used inside RACE, adapted from vision."""
    def __init__(self, d_k, K, L, M, device="cpu", share_planes=False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer("planes_T", planes.view(L * K, d_k).T)
        else:
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)
            self.register_buffer("planes_T", planes)

        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer("protos_T", corners.T)

        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps=1e-6):
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            projK = Kh2 @ self.planes_T
            projQ = Qh2 @ self.planes_T
        else:
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            projK = torch.einsum("mbtd,mds->mbts", Kh2, self.planes_T)
            projQ = torch.einsum("mbtd,mds->mbts", Qh2, self.planes_T)

            projK = projK.contiguous().view(M * BH, T, self.L * self.K)
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsK  = F.softmax(logitsK, dim=-1)
        probsQ  = F.softmax(logitsQ, dim=-1)

        probsK_S = probsK.contiguous().view(N, T, S)
        probsQ_S = probsQ.contiguous().view(N, T, S)

        b_sum = probsK_S.transpose(1, 2).bmm(V2)      # [N,S,dk]
        A     = probsK_S.sum(dim=1)                   # [N,S]
        E     = b_sum / (A.unsqueeze(-1) + eps)       # [N,S,dk]

        out2 = probsQ_S.bmm(E)                        # [N,T,dk]
        out  = out2.view(M, B, H, T, dk).permute(0, 1, 2, 3, 4)
        return out

class RACEAttention(nn.Module):
    def __init__(self, d, h, drop, K, L, M, qkv_bias=False, device="cpu"):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.M = h, d // h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(self.dk, K, L, M, device=device)

    def forward(self, x, mask):
        B, T, d = x.shape
        h, dk, M = self.h, self.dk, self.M

        Q = self.q(x).view(B, T, h, dk)
        K = self.k(x).view(B, T, h, dk)
        V = self.v(x).view(B, T, h, dk)

        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)
            Q, K, V = Q * m, K * m, V * m

        def pack(z):
            return z.unsqueeze(0).expand(M, -1, -1, -1, -1)

        out_m = self.ace(pack(K), pack(V), pack(Q))   # [M,B,H,T,dk]
        out   = out_m.mean(dim=0)                     # [B,H,T,dk]
        out   = out.permute(0, 2, 1, 3).contiguous().view(B, T, h * dk)
        return self.drop(self.o(out))

# ---- FAVOR+ (Performer) ----
def favorplus_features(x, proj, eps=1e-6):
    xw = torch.einsum("bhtd,hmd->bhtm", x, proj)
    xw = xw - xw.max(dim=-1, keepdim=True).values
    exp_part  = torch.exp(xw)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    base      = torch.exp(-0.5 * x_norm_sq)
    return exp_part * base + eps

class FavorPlusAttention(nn.Module):
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)
        self.eps = 1e-6

    def forward(self, x, mask=None):
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            Ks   = Ks * keep
            V    = V  * keep

        phiQ = favorplus_features(Qs, self.proj, eps=self.eps) / math.sqrt(m)
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)

        if mask is not None:
            keep_m = mask[:, None, :, None].to(phiK.dtype)
            phiK   = phiK * keep_m

        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den

        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        merged = self.drop(merged)
        return self.o(merged)

# ---- Linear attention (ELU kernel) ----
class LinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def kernel(self, x):
        return F.elu(x) + 1

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        H, D = self.num_heads, self.head_dim

        Q = self.W_query(x).view(B, T, H, D).transpose(1, 2)
        K = self.W_key(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_value(x).view(B, T, H, D).transpose(1, 2)

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Q = self.kernel(Q)
        K = self.kernel(K)

        KV = torch.einsum("bhtd,bhte->bhde", K, V)  # [B,H,D,D]
        K_sum = K.sum(dim=2)                       # [B,H,D]

        Z = torch.einsum("bhtd,bhd->bht", Q, K_sum) + self.eps
        context = torch.einsum("bhtd,bhde->bhte", Q, KV)
        out = context / Z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out = self.dropout(out)
        return self.out_proj(out)

# ---- Linformer attention ----
class LinformerAttention(nn.Module):
    def __init__(self, d, dropout, num_heads, qkv_bias, k_proj_dim, max_seq_len):
        super().__init__()
        assert d % num_heads == 0
        self.h  = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        self.W_query = nn.Linear(d, d, bias=qkv_bias)
        self.W_key   = nn.Linear(d, d, bias=qkv_bias)
        self.W_value = nn.Linear(d, d, bias=qkv_bias)

        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        assert T <= self.max_seq_len
        h, dk, k = self.h, self.dk, self.k_proj_dim

        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        if mask is not None:
            keep = mask[:, None, :, None].to(Q.dtype)
            K = K * keep
            V = V * keep

        Ek = self.E_k[:T]  # (T,k)
        Ev = self.E_v[:T]

        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)  # [B,h,k,dk]
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        out = self.dropout(out)
        return self.out_proj(out)

# ==================================================
# 6) Transformer blocks (one per baseline)
# ==================================================
class SoftmaxBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = MultiHeadAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att  = AngularAttention(d, h, drop, qkv_bias)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class RACEBlock(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        self.att = RACEAttention(
            d=d, h=h, drop=drop,
            K=cfg["K"], L=cfg["L"], M=cfg["M"],
            qkv_bias=qkv_bias, device=device,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

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

    def forward(self, x, mask):
        B, T, D = x.shape
        H = self.num_heads
        dim = self.head_dim

        Q = self.q_proj(x).view(B, T, H, dim)  # [B, T, H, dim]
        K = self.k_proj(x).view(B, T, H, dim)
        V = self.v_proj(x).view(B, T, H, dim)
        
        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1).to(Q.dtype)  # [B, T, 1, 1]
            Q, K, V = Q * m, K * m, V * m

        Q = Q.transpose(1, 2)  # [B, H, T, dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

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
            m = torch.zeros_like(probsK)
            m.scatter_(-1, top_idx, 1.0)

            # keep only top_k entries
            probsK = probsK * m

            # renormalize over R so it still sums to 1
            probsK = probsK / (probsK.sum(dim=-1, keepdim=True) + 1e-6)

        # 6) Collapse (L, R) -> S buckets
        probsK_S = probsK.view(N, T, S)                    # [N, T, S]
        assign_probs = probsK_S.view(B, H, T, S)           # [B, H, T, S]

        # 7) Build bucketed K and V by weighted sums over time
        bucket_K = torch.einsum('bhts,bhtd->bhsd', assign_probs, K)  # [B, H, S, dim]
        bucket_V = torch.einsum('bhts,bhtd->bhsd', assign_probs, V)  # [B, H, S, dim]

        # Normalize buckets: average instead of sum
        A = assign_probs.sum(dim=2)        # [B, H, S]
        A = A.unsqueeze(-1)                # [B, H, S, 1]
        bucket_K = bucket_K / (A + 1e-6)
        bucket_V = bucket_V / (A + 1e-6)

        # 8) Query interacts with bucketed keys
        scores = torch.einsum("bhtd,bhsd->bhts", Q, bucket_K) / math.sqrt(dim)  # [B,H,T,S]
        probs_attn = F.softmax(scores, dim=-1)  # [B, H, T, S]

        # 9) Mix bucketed values
        out = torch.einsum("bhts,bhsd->bhtd", probs_attn, bucket_V)  # [B, H, T, dim]

        # 10) Merge heads: [B, H, T, dim] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.drop(out)


class MACHBlock(nn.Module):
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.att   = MACHAttention(d_k=cfg["embed_dim"], K_bits=cfg["K"], heads=cfg["num_heads"], L=cfg["L"], 
                                   dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"], top_k=2, device=device)
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        self.att  = LinearAttention(
            d_in=d, d_out=d, dropout=drop, num_heads=h, qkv_bias=qkv_bias
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class LinformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        h = cfg["num_heads"]
        k_proj_dim = 128
        self.att  = LinformerAttention(
            d=d,
            dropout=drop,
            num_heads=h,
            qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim,
            max_seq_len=cfg["max_len"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class PerformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        h = cfg["num_heads"]
        drop = cfg["drop_rate"]
        self.att = FavorPlusAttention(
            d=d,
            h=h,
            m_features=cfg["m_features"],
            drop=drop,
            qkv_bias=cfg["qkv_bias"],
            seed=cfg["favor_seed"],
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], d),
        )
        self.drop  = nn.Dropout(drop)

    def forward(self, x, mask):
        h = x
        x = self.norm1(x)
        x = self.att(x, mask)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
    
# ==================================================
# 7) Text Transformer classifier (ViT-style structure)
# ==================================================
class TextTransformerClassifier(nn.Module):
    def __init__(self, cfg, attn_type: str):
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg["vocab_size"]
        max_len    = cfg["max_len"]
        d          = cfg["embed_dim"]

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop    = nn.Dropout(cfg["drop_rate"])

        if attn_type == "softmax":
            Block = SoftmaxBlock
        elif attn_type == "race":
            Block = lambda c: RACEBlock(c, device=DEVICE)
        elif attn_type == "angular":
            Block = AngularBlock
        elif attn_type == "linear":
            Block = LinearBlock
        elif attn_type == "linformer":
            Block = LinformerBlock
        elif attn_type == "performer":
            Block = PerformerBlock
        elif attn_type == "mach":
            Block = lambda c: MACHBlock(c, device=DEVICE)
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        self.layers = nn.ModuleList(
            [Block(cfg) for _ in range(cfg["num_layers"])]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x, mask):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.layers:
            h = blk(h, mask)
        h = self.norm(h)
        # CLS-style: use position 0
        logits = self.head(h[:, 0])
        return logits
