import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaAdapter(nn.Module):
    def __init__(self, d_model=256, bottleneck_dim=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        #self.dropout = nn.Dropout(0.1)
        # LayerNorm 对 text_feats 做归一化
        self.text_norm = nn.LayerNorm(d_model)
        # LayerNorm 对 visual_support 做归一化
        self.vis_norm = nn.LayerNorm(d_model)
        # LayerNorm 对最终融合特征归一化
        self.out_norm = nn.LayerNorm(d_model)

        # gating block
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        # MLP layers for query/key transform (Meta-Adapter 论文中 W1, W2)
        self.W1 = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.W2 = nn.Linear(d_model, bottleneck_dim, bias=False)

        self.scale = (bottleneck_dim // num_heads) ** -0.5

    def forward(self, text_feats, visual_support):
        """
        text_feats: [B, L_t, D]   -> 类别或token级别文本特征
        visual_support: [B, L_v, D] -> few-shot视觉支持样本
        """
        # === 1. 归一化文本特征 ===
        text_feats_norm = self.text_norm(text_feats)  # [B, L_t, D]
        visual_support_norm = self.vis_norm(visual_support)

        # === 2. Cross-attention ===
        Q = self.W1(text_feats_norm)  # [B, L_t, bottleneck_dim]
        K = self.W2(visual_support_norm)  # [B, L_v, bottleneck_dim]

        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # [B, L_t, L_v]
        attn = F.softmax(attn, dim=-1)

        # 聚合视觉特征
        agg_feat = torch.bmm(attn, visual_support_norm)  # [B, L_t, D]

        # === 3. gating modulation ===
        gate_val = self.gate(text_feats_norm)  # [B, L_t, 1]
        fused = text_feats + gate_val * agg_feat

        # === 4. 输出归一化 ===
        refined = self.out_norm(fused)

        return refined

# class MetaAdapter(nn.Module):
#     def __init__(self, d_model=256, vis_in_dim=None):
#         super().__init__()
#         self.d_model = d_model
#
#         # 如果视觉特征维度不同，需要投影到 d_model
#         if vis_in_dim is not None and vis_in_dim != d_model:
#             self.vis_proj = nn.Linear(vis_in_dim, d_model)
#         else:
#             self.vis_proj = nn.Identity()
#
#         self.text_norm = nn.LayerNorm(d_model)
#         self.vis_norm = nn.LayerNorm(d_model)
#         self.out_norm = nn.LayerNorm(d_model)
#
#         # FiLM parameters
#         self.film_gamma = nn.Linear(d_model, d_model)
#         self.film_beta  = nn.Linear(d_model, d_model)
#
#         self.gate_fc = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, text_feats, visual_support):
#         """
#         text_feats:     [B, L_t, D]
#         visual_support: [B, L_v, C_in] → will be projected to d_model
#         """
#         # 视觉特征投影
#         visual_support = self.vis_proj(visual_support)
#
#         # ===== 1. Normalize =====
#         text_norm = self.text_norm(text_feats)
#         vis_norm  = self.vis_norm(visual_support)
#
#         # ===== 2. Few-shot prototype =====
#         proto = vis_norm.mean(dim=1, keepdim=True)  # [B, 1, D]
#
#         # ===== 3. FiLM modulation =====
#         gamma = self.film_gamma(proto)
#         beta  = self.film_beta(proto)
#         t_mod = gamma * text_norm + beta
#
#         # ===== 4. gate =====
#         gate = self.gate_fc(proto)
#
#         fused = text_feats + gate * t_mod
#
#         return self.out_norm(fused)

