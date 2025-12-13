import torch 
import torch.nn as nn
import itertools
import math
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
from torch.nn.attention import sdpa_kernel, SDPBackend

class PatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = nn.Conv2d(cfg["num_channels"], cfg["embed_dim"], kernel_size=cfg["patch_size"], stride=cfg["patch_size"])

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.dropout_p = dropout

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape   
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        Q, K, V = [t.to(dtype=torch.float16) for t in (Q, K, V)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,      # we keep dropout on the output like before
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.dropout(out)
        out = out.to(self.out_proj.weight.dtype)
        return self.out_proj(out)
    
class TransformerArchitecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(cfg["embed_dim"])
        self.self_attention = MultiHeadAttention(d_in=cfg["embed_dim"], d_out=cfg["embed_dim"], dropout=cfg["drop_rate"], num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"])
        self.layer_norm_2 = nn.LayerNorm(cfg["embed_dim"])
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )

    def forward(self, x):
        residual_1 = x
        attention_output = self.self_attention(self.layer_norm_1(x))
        x = attention_output + residual_1
        residual_2 = x
        mlp_output = self.multi_layer_perceptron(self.layer_norm_2(x))
        x = mlp_output + residual_2
        return x

class BatchedACE(nn.Module):
    """
    Non-causal BatchedACE with optional shared planes.
    Inputs:
      Khf, Vhf, Qhf : [M, B, T, H, d_k]
    """
    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            # Shared planes [L, K, d_k] --> [d_k, (L*K)]
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)   # [d_k, L*K]
        else:
            # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)           # [M, d_k, L*K]
            self.register_buffer('planes_T', planes)

        # Prototypes (corners of {-1,+1}^K): [K, R]
        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)                        # [K, R]

        # learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    
    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        # Khf, Vhf, Qhf: [M, B, T, H, d_k]
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 20.0)

        if self.share_planes:
            # Collapse M·B·H → N
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            # Projections to L*K
            projK = Kh2 @ self.planes_T                                     # [N,T,L*K]
            projQ = Qh2 @ self.planes_T                                     # [N,T,L*K]
        else:
            # Keep ensembles separate; collapse only B·H
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            # One GEMM per ensemble
            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)        # [M,BH,T,L*K]
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)
            # Merge M,BH → N
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)       # [N,T,L*K]
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        # Reshape to [N,T,L,K] and soft-hash → probs over R buckets
        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        probsK  = F.softmax(logitsK, dim=-1)                                   # [N,T,L,R]
        probsQ  = F.softmax(logitsQ, dim=-1)                                   # [N,T,L,R]

        # -------- Non-causal bucket summaries over the full sequence --------
        # Collapse buckets L,R → S
        probsK_S = probsK.contiguous().view(N, T, S)                           # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)                           # [N,T,S]

        # Weighted sums across time:
        #   b_sum = probsK^T @ V   → [N,S,dk]
        b_sum = probsK_S.transpose(1, 2).bmm(V2)                               # [N,S,dk]
        #   A = sum_t probsK_t     → [N,S]
        A = probsK_S.sum(dim=1)                                                # [N,S]
        #   E = b_sum / (A + eps)  → [N,S,dk]$Ginger@0907&

        E = b_sum / (A.unsqueeze(-1) + eps)                                    # [N,S,dk]

        # Query lookup per time (no prefix): [N,T,S] @ [N,S,dk] → [N,T,dk]
        out2 = probsQ_S.bmm(E)                                                 # [N,T,dk]
        # Unflatten back to [M,B,T,H,dk]
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)                 # [M,B,T,H,dk]
        return out
    
class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,
                 num_heads, L, K, N_M, qkv_bias=False, device='cpu'):
        super().__init__()
        assert d_in % num_heads == 0
        self.H   = num_heads
        self.d_k = d_in // num_heads
        self.M   = N_M
        self.L = L
        self.P = K

        self.q_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.out    = nn.Linear(d_in, d_out)
        self.drop   = nn.Dropout(dropout)
        self.ace = BatchedACE(self.d_k, K, L, N_M, device=device)

    def forward(self, x):
        B, T, _ = x.shape
        H, d_k, M = self.H, self.d_k, self.M

        # 1) project & reshape for ACE
        Q = self.q_proj(x).view(B, T, H, d_k)
        K = self.k_proj(x).view(B, T, H, d_k)
        V = self.v_proj(x).view(B, T, H, d_k)

        # shape --> [M, B, T, H, d_k] by explicit unsqueeze
        def pack(Z):
            Zm = Z.unsqueeze(0).expand(M, -1, -1, -1, -1)
            return Zm

        Khf = pack(K)
        Vhf = pack(V)
        Qhf = pack(Q)

        # 2) run ACE
        out_hm = self.ace(Khf, Vhf, Qhf)  # [M,B,T,H,d_k]

        # 3) average ensembles & merge heads
        out = out_hm.mean(dim=0)          # [B,T,H,d_k]
        out = out.permute(0,2,1,3).reshape(B, T, H * d_k)

        # 4) final proj + dropout
        return self.drop(self.out(out))
    

class MACHAttention(nn.Module):
    def __init__(self, d_k, heads, K_bits, L, dropout, qkv_bias=False, device='cuda'):
        super().__init__()
        self.d_k = d_k              # model dim
        self.num_heads = heads
        self.head_dim = d_k // heads

        self.K_bits = K_bits
        self.L = L
        self.R = 1 << K_bits
        self.S = self.L * self.R

        # Random planes: [L, K_bits, head_dim] -> [head_dim, L*K_bits]
        planes = torch.randn(L, K_bits, self.head_dim, device=device)
        self.register_buffer(
            'planes_T',
            planes.view(L * K_bits, self.head_dim).T  # [head_dim, L*K_bits]
        )

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

        # 6) Collapse (L, R) -> S buckets
        probsK_S = probsK.view(N, T, S)                    # [N, T, S]
        assign_probs = probsK_S.view(B, H, T, S)           # [B, H, T, S]

        # 7) Build bucketed K and V by weighted sums over time
        bucket_K = torch.einsum('bhts,bhtd->bhsd', assign_probs, K)  # [B, H, S, dim]
        bucket_V = torch.einsum('bhts,bhtd->bhsd', assign_probs, V)  # [B, H, S, dim]

        # 8) Query interacts with bucketed keys
        scores = torch.einsum("bhtd,bhsd->bhts", Q, bucket_K) / math.sqrt(dim)  # [B,H,T,S]
        probs_attn = F.softmax(scores, dim=-1)  # [B, H, T, S]

        # 9) Mix bucketed values
        out = torch.einsum("bhts,bhsd->bhtd", probs_attn, bucket_V)  # [B, H, T, dim]

        # 10) Merge heads: [B, H, T, dim] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.drop(out)


class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att   = RACEAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["num_heads"], qkv_bias=cfg["qkv_bias"],
            L=cfg["L"], K=cfg["K"], N_M=cfg["M"], device=device
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class MACHBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att   = MACHAttention(d_k=cfg["embed_dim"], K_bits=cfg["K"], heads=cfg["num_heads"], L=cfg["L"], 
                                   dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"], device=device)
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
            nn.GELU(),
            nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

def favorplus_features(x, proj, eps=1e-6):
    """
    FAVOR+ positive random features for softmax kernel.
    x:    [B,H,T,D]
    proj: [H,M,D]  (one matrix per head; rows ~ N(0, I))
    ->    [B,H,T,M]  (non-negative)
    """
    # x @ W^T  -> [B,H,T,M]
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)

    # stabilize across feature dimension
    xw = xw - xw.max(dim=-1, keepdim=True).values

    # exp( xW^T - ||x||^2/2 )
    exp_part  = torch.exp(xw)                         # [B,H,T,M]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)    # [B,H,T,1]
    base      = torch.exp(-0.5 * x_norm_sq)           # [B,H,T,1]
    return exp_part * base + eps                      # strictly positive


class FavorPlusAttention(nn.Module):
    """
    Non-causal FAVOR+ (Performer) attention (softmax kernel via positive RF).
    - Pad-mask aware (mask: 1=keep, 0=pad).
    - Saves pre-projection context in self.last_ctx (B,T,d) and per-head in self.last_ctx_heads (B,H,T,dk).
    """
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0, "Embedding dim must be divisible by num_heads"
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        # Draw Gaussian projection matrices per head: [H,M,Dk]
        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.nn.init.orthogonal_(torch.randn(h, m_features, self.dk))
        self.register_buffer("proj", proj)             # no grad; moves with device

        # For inspection/plots
        self.ctx = None
        self.eps = 1e-6

    def forward(self, x):
        """
        x:    (B, T, d)
        mask: (B, T) with 1/True = keep, 0/False = pad
        return: (B, T, d)
        """
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        # Projections -> (B,H,T,dk)
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Scale like softmax attention: exp(q·k / sqrt(dk)) ≡ use q/√dk inside features
        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)


        # Positive random features
        phiQ = favorplus_features(Qs, self.proj, eps=self.eps)/ math.sqrt(m)   # [B,H,T,M]
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)  # [B,H,T,M]


        # Global (non-causal) aggregation over time
        # KV   = sum_t phiK_t^T ⊗ V_t  -> (B,H,M,dk)
        # Ksum = sum_t phiK_t          -> (B,H,M)
        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        # Per-query readout
        # num = phiQ @ KV   -> (B,H,T,dk)
        # den = phiQ · Ksum -> (B,H,T,1)
        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den                          # (B,H,T,dk)

        # Save pre-projection context for visualization
        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        self.ctx = merged

        # Standard output path
        merged = self.drop(merged)
        return self.o(merged)
    

class PerformerBlock(nn.Module):
    """
    Residual block with FAVOR+ attention + FFN, mirroring your LinearBlock.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = FavorPlusAttention(
            d=cfg["embed_dim"],
            h=cfg["num_heads"],
            m_features=cfg.get("m_features", 256),
            drop=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False),
            seed=cfg.get("favor_seed", None),
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm + attention
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


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
        # φ(x): positive-valued kernel feature map
        return F.elu(x) + 1  # [B, H, T, D]

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D]
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Apply kernel φ
        Q = self.kernel(Q)  # [B, H, T, D]
        K = self.kernel(K)  # [B, H, T, D]

        # Compute KV^T: [B, H, D, D]
        KV = torch.einsum('bhtd,bhte->bhde', K, V)  # [B, H, D, D]

        # Compute normalization factor: Z = Q * sum(K)
        K_sum = K.sum(dim=2)  # [B, H, D]
        Z = torch.einsum('bhtd,bhd->bht', Q, K_sum) + self.eps  # [B, H, T]

        # Compute output: Q @ (KV)
        context = torch.einsum('bhtd,bhde->bhte', Q, KV)  # [B, H, T, D]
        out = context / Z.unsqueeze(-1)  # [B, H, T, D]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, H*D]
        return self.out_proj(out)

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = LinearAttention(
            d_in=cfg["embed_dim"], d_out=cfg["embed_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["num_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"],cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"],cfg["embed_dim"])
                        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x
    
class AngularAttention(nn.Module):
    def __init__(self, d, h, drop, qkv_bias=False):
        super().__init__()
        self.h, self.dk = h, d//h
        self.q = nn.Linear(d,d, bias=qkv_bias)
        self.k = nn.Linear(d,d, bias=qkv_bias)
        self.v = nn.Linear(d,d, bias=qkv_bias)
        self.o = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,T,_ = x.shape
        Q = F.normalize(self.q(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        K = F.normalize(self.k(x).view(B,T,self.h,self.dk).transpose(1,2), dim=-1)
        V = self.v(x).view(B,T,self.h,self.dk).transpose(1,2)
        sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores = 1 - torch.acos(sim)/math.pi
        W = scores.clamp(min=1e-6).pow(8)
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.drop(W)
        out = (W @ V).transpose(1,2).contiguous().view(B,T,self.h*self.dk)
        return self.o(out)
    
class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = AngularAttention(d=cfg["embed_dim"], h=cfg["num_heads"], drop=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])

        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x
    
class LinformerAttention(nn.Module):
    """
    Linformer-style attention: project K,V along sequence length T -> k (k << T),
    then do standard scaled dot-product attention with softmax over k.

    Shapes:
      x: (B, T, d_in)
      returns: (B, T, d_out)
    """
    def __init__(
        self,
        d: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        k_proj_dim: int,      # low-rank sequence dim
        max_seq_len: int    # allocate E up to this T, slice at runtime
    ):
        super().__init__()
        assert d % num_heads == 0, "d_out must be divisible by num_heads"
        self.h = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        # token projections
        self.W_query = nn.Linear(d,  d, bias=qkv_bias)
        self.W_key   = nn.Linear(d,  d, bias=qkv_bias)
        self.W_value = nn.Linear(d,  d, bias=qkv_bias)

        # learnable sequence projections E_k, E_v: [T_max, k]
        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))

        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        h, dk, k = self.h, self.dk, self.k_proj_dim

        # Linear projections -> (B, h, T, dk)
        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(  x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Sequence down-projection (T -> k) using E_k/E_v sliced to current T
        Ek = self.E_k[:T]  # (T, k)
        Ev = self.E_v[:T]  # (T, k)

        # K_proj, V_proj: (B, h, k, dk)
        # Contract over sequence axis
        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        # Scaled dot-product attention over compressed length k
        # scores: (B, h, T, k)
        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        # Context: (B, h, T, dk)
        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)

        # Merge heads -> (B, T, d_out)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.out_proj(self.dropout(out))


class LinformerBlock(nn.Module):
    """
    Drop-in analogue of your LinearBlock but using LinformerAttention.
    Non-causal, no kernel; just K,V low-rank sequence projection.
    """
    def __init__(self, cfg):
        super().__init__()
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        k_proj_dim = 128

        self.att  = LinformerAttention(
            d=cfg["embed_dim"], dropout=drop, num_heads=cfg["num_heads"], qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim, max_seq_len=cfg["num_patches"] + 1
        )
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["embed_dim"], cfg["mlp_dim"]),
                        nn.GELU(),
                        nn.Linear(cfg["mlp_dim"], cfg["embed_dim"]),
                     )
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        # Attn sublayer
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN sublayer
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

class VisionTransformer(nn.Module):
    def __init__(self, cfg, attn_type, device='cuda'):
        super().__init__()
        self.patch_embedding = PatchEmbedding(cfg)

        G = cfg["img_size"] // cfg["patch_size"]
        assert G * cfg["patch_size"] == cfg["img_size"], "img_size must be divisible by patch_size"
        num_patches = G * G
        d = cfg["embed_dim"]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # pick block
        if attn_type == "softmax":
            AttnBlock = TransformerArchitecture
        elif attn_type == "mach":
            AttnBlock = MACHBlock
        elif attn_type == "race":
            AttnBlock = lambda c: RACEBlock(c, device)
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "linear":
            AttnBlock = LinearBlock
        elif attn_type == "linformer":
            AttnBlock = LinformerBlock
        elif attn_type == "performer":
            AttnBlock = PerformerBlock
        else:
            raise ValueError("Unsupported attention type")

        self.transformer_layers = nn.Sequential(
            *[AttnBlock(cfg) for _ in range(cfg["transformer_units"])]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, cfg["mlp_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["mlp_dim"], cfg["num_classes"])
        )

    def forward(self, x):
        x = self.patch_embedding(x)                 # [B, N, d], N=G*G
        B, N, d = x.shape
        cls = self.cls_token.expand(B, -1, -1)      # [B,1,d]
        x = torch.cat([cls, x], dim=1)              # [B, N+1, d]
        x = x + self.pos_embed[:, :x.size(1), :]    # safe slice
        x = self.transformer_layers(x)
        x = x[:, 0]                                 # CLS
        return self.mlp_head(x)