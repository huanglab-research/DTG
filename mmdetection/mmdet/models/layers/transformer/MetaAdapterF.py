import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MetaAdapterF(nn.Module):
    def __init__(self, d_model=256, bottleneck_dim=64, num_heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.Wv = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.scale = (bottleneck_dim // num_heads) ** -0.5
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, query, visual_feats):
        """
        query: [B, Nq, D]
        visual_feats: [B, Lv, D] — encoder输出的特征（key/value）
        """
        q = self.norm_q(query)
        v = self.norm_v(visual_feats)
        Q = self.Wq(q)
        V = self.Wv(v)
        attn = torch.matmul(Q, V.transpose(-1, -2)) * self.scale  # [B, Nq, Lv]
        attn = F.softmax(attn, dim=-1)
        context = torch.bmm(attn, visual_feats)  # [B, Nq, D]

        gate = self.gate(q)  # [B, Nq, 1]
        fused = query + gate * context
        return self.out_norm(fused)
