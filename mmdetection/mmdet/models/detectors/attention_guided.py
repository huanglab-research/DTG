import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGuidedModule(nn.Module):
    """
    使用 decoder cross-attention 引导视觉特征聚焦目标区域
    """
    def __init__(self, smooth=True):
        super().__init__()
        self.smooth = smooth
        if smooth:
            # 可选卷积，用于平滑注意力图
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.norm = nn.BatchNorm2d(1)
            self.act = nn.Sigmoid()

    def forward(self, visual_feats, attn_weights):
        """
        visual_feats: [B, C, H, W]
        attn_weights: [B, heads, queries, HW]
        """
        B, C, H, W = visual_feats.shape

        # 1. 平均 head 与 query
        attn_map = attn_weights.mean(1).mean(1)  # [B, HW]

        # 2. reshape 成空间图
        attn_map = attn_map.view(B, 1, H, W)

        # 3. 归一化
        attn_map = attn_map / (attn_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)

        # 4. 可选平滑
        if self.smooth:
            attn_map = self.act(self.norm(self.conv(attn_map)))

        # 5. 视觉特征加权
        visual_feats_refined = visual_feats * attn_map

        return visual_feats_refined, attn_map
