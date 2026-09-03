import torch
import torch.nn as nn
import torch.nn.functional as F


class DoGForeground(nn.Module):
    def __init__(self,
                 sigma1=1.0,
                 sigma2=2.0,
                 gamma=0.05,
                 num_groups=32):
        super().__init__()
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.norm = None
        self.num_groups = num_groups

        # lazy init
        self.channel_gate = None


    def gaussian_kernel(self, kernel_size, sigma, device):
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def apply_gaussian(self, x, sigma):
        # x: [B, 1, H, W]
        k = int(2 * round(3 * sigma) + 1)
        kernel = self.gaussian_kernel(k, sigma, x.device)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2)

    def forward(self, F):
        B, C, H, W = F.shape

        if H < 16 or W < 16:
            if self.norm is None:
                self.norm = nn.GroupNorm(self.num_groups, C).to(F.device)
            return self.norm(F), torch.zeros(B, H, W, device=F.device)

        if self.norm is None:
            self.norm = nn.GroupNorm(self.num_groups, C).to(F.device)

        # -------- lazy init channel gate --------
        if self.channel_gate is None:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B,C,1,1]
                nn.Conv2d(C, C, 1),  # channel-wise
                nn.Sigmoid()
            ).to(F.device)

        # spatial projection
        x = F.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # -------- DoG --------
        blur1 = self.apply_gaussian(x, self.sigma1)
        blur2 = self.apply_gaussian(x, self.sigma2)
        dog = blur1 - blur2  # [B,1,H,W]

        S = dog.squeeze(1)  # [B,H,W]

        # -------- normalize S --------
        S_flat = S.view(B, -1)
        S_min = S_flat.min(dim=1)[0].view(B, 1, 1)
        S_max = S_flat.max(dim=1)[0].view(B, 1, 1)

        S = (S - S_min) / (S_max - S_min + 1e-6)
        S = S - S.mean(dim=(1, 2), keepdim=True)

        # =====================================================
        # 改动 2：channel-aware gamma
        # =====================================================
        gamma_c = self.channel_gate(F)  # [B,C,1,1]

        # =====================================================
        # 改动 3：DoG gating（不是直接乘）
        # =====================================================
        spatial_gate = torch.sigmoid(self.gamma * S.unsqueeze(1))
        gate = spatial_gate * gamma_c  # [B,C,H,W]

        F_refined = F + F * gate
        F_refined = self.norm(F_refined)

        return F_refined, S




