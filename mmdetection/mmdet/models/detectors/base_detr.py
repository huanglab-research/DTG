# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class DetectionTransformer(BaseDetector, metaclass=ABCMeta):
    r"""Base class for Detection Transformer.

    In Detection Transformer, an encoder is used to process output features of
    neck, then several queries interact with the encoder features using a
    decoder and do the regression and classification with the bounding box
    head.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

        # === 背景库初始化 ===
        bg_path = "/home/hl/my_data/zyr/ETS/bg_library.npy"

        # 动态检测 backbone 输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            feat_dim = feats[-1].shape[1] if isinstance(feats, (list, tuple)) else feats.shape[1]

        # 加载背景库
        self.load_background_library(bg_path, feat_dim)

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    # def extract_feat(self, batch_inputs: torch.Tensor, batch_masks=None) -> list:
    #     """
    #     提取特征（训练时自动执行 AdaIN 背景扰动）。
    #     Args:
    #         batch_inputs (Tensor): 输入图像 (B, 3, H, W)
    #         batch_masks (Tensor, optional): 前景掩码 (B, 1, H, W)，用于控制扰动范围
    #     Returns:
    #         list[Tensor]: 特征图序列 (B, C, H, W)
    #     """
    #     # === Step 1: 提取原始特征 ===
    #     x = self.backbone(batch_inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #
    #     # 确保输出是 list
    #     if not isinstance(x, (list, tuple)):
    #         x = [x]
    #
    #     # === Step 2: 若在训练模式并提供 mask，则执行背景扰动 ===
    #     if self.training and batch_masks is not None:
    #         perturbed_feats = []
    #         for feat in x:
    #             if isinstance(feat, tuple):
    #                 feat_tensor, hw_shape = feat
    #             else:
    #                 feat_tensor = feat
    #
    #             # 重要：detach + clone + requires_grad_(True)
    #             feat_tensor = feat_tensor.detach().clone().requires_grad_(True)
    #
    #             # 估计 num_shot（控制扰动幅度）
    #             nonzero = batch_masks.view(batch_masks.shape[0], -1).sum(dim=1)
    #             avg_fg = (nonzero > 0).sum().item()
    #             num_shot = max(1, avg_fg)
    #
    #             # 执行背景扰动
    #             feat_tensor = self.adain_perturb(feat_tensor, mask=batch_masks, num_shot=num_shot)
    #             perturbed_feats.append(feat_tensor)
    #
    #         x = perturbed_feats
    #
    #     return x
    #
    # import torch
    # import torch.nn.functional as F

    @torch.no_grad()
    def match_bg_stats(self, target_feats: torch.Tensor, top_k: int = 3, num_shot: int = 1):
        """
        匹配目标特征与背景库的均值方差，用于扰动生成。
        target_feats: [B, C, H, W] 或 [B, C]
        返回: (mu_t, sigma_t)
        """
        device = target_feats.device

        # --- 1. 获取 target embed ---
        if target_feats.ndim == 4:
            target_embed = target_feats.mean(dim=[2, 3])  # [B, C]
        else:
            target_embed = target_feats  # [B, C]
        target_embed = F.normalize(target_embed, dim=1)

        # --- 2. 保证 bg_embed 维度匹配 ---
        bg_embed = self.bg_embed.to(device)
        if bg_embed.shape[1] != target_embed.shape[1]:
            # 初始化投影层（仅一次）
            if not hasattr(self, 'bg_adapt_proj') or \
                    self.bg_adapt_proj.weight.shape[0] != target_embed.shape[1]:
                self.bg_adapt_proj = torch.nn.Linear(
                    bg_embed.shape[1], target_embed.shape[1], bias=False
                ).to(device)
            bg_embed = self.bg_adapt_proj(bg_embed)

        bg_embed = F.normalize(bg_embed, dim=1)

        # --- 3. 相似度匹配 ---
        sim = torch.matmul(target_embed, bg_embed.T)  # [B, N_bg]
        topk_sim, topk_idx = torch.topk(sim, k=min(top_k, sim.shape[1]), dim=1)

        # --- 4. 计算匹配加权平均的 mu / sigma ---
        bg_mu = self.bg_mu.to(device)
        bg_sigma = self.bg_sigma.to(device)

        if bg_mu.shape[1] != target_embed.shape[1]:
            # 需要同样的投影
            bg_mu = self.bg_adapt_proj(bg_mu)
            bg_sigma = self.bg_adapt_proj(bg_sigma)

        mu_t, sigma_t = [], []
        for b in range(topk_idx.size(0)):
            idx = topk_idx[b]  # [K]
            weights = F.softmax(topk_sim[b], dim=0)
            mu_t.append(torch.sum(bg_mu[idx] * weights[:, None], dim=0))
            sigma_t.append(torch.sum(bg_sigma[idx] * weights[:, None], dim=0))

        mu_t = torch.stack(mu_t, dim=0)
        sigma_t = torch.stack(sigma_t, dim=0)
        sigma_t = torch.clamp(sigma_t, min=1e-4, max=2.0)

        return mu_t, sigma_t

    # def extract_feat_with_perturb(self, batch_inputs, batch_masks=None):
    #     """
    #     对输入特征进行背景扰动增强。
    #     原始特征不反传（detach），扰动特征参与训练。
    #     """
    #     if isinstance(batch_inputs, (list, tuple)):
    #         feats = [f for f in batch_inputs]
    #     else:
    #         feats = self.extract_feat(batch_inputs)  # 原始特征图
    #
    #     perturbed_feats = []
    #
    #     for feat in feats:
    #         if isinstance(feat, tuple):
    #             feat_tensor, hw_shape = feat
    #         else:
    #             feat_tensor = feat
    #
    #         # Step 1: detach，防止梯度回传两次
    #         feat_tensor = feat_tensor.detach()
    #
    #         # Step 2: 估计 num_shot
    #         if batch_masks is not None:
    #             nonzero = batch_masks.view(batch_masks.shape[0], -1).sum(dim=1)
    #             avg_fg = (nonzero > 0).sum().item()
    #             num_shot = max(1, avg_fg)
    #         else:
    #             num_shot = 1
    #
    #         # Step 3: 应用 AdaIN 扰动
    #         perturbed_feat = self.adain_perturb(feat_tensor, mask=batch_masks, num_shot=num_shot)
    #         perturbed_feats.append(perturbed_feat)
    #
    #     return perturbed_feats

    def adain_perturb(self, feat: torch.Tensor, mask: torch.Tensor, img: torch.Tensor = None,
                      num_shot: int = 1, save_dir="./visuals", prefix="sample", save_vis=True):
        """
        对特征进行 AdaIN 风格扰动，仅作用于背景区域。
        """
        assert feat.ndim == 4, f"Expected 4D feat, got {feat.shape}"
        device = feat.device
        B, C, H, W = feat.shape
        os.makedirs(save_dir, exist_ok=True)

        # === 1. 调整 mask 尺寸 ===
        if mask.shape[2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='nearest')
        mask_fg = (mask > 0).float()
        mask_bg = 1 - mask_fg
        eps = 1e-6

        # === 2. 计算前景统计（用于归一化） ===
        mu_fg = (feat * mask_fg).sum(dim=[2, 3]) / (mask_fg.sum(dim=[2, 3]) + eps)
        var_fg = ((feat - mu_fg[:, :, None, None]) ** 2 * mask_fg).sum(dim=[2, 3]) / (
                mask_fg.sum(dim=[2, 3]) + eps
        )
        sigma_fg = torch.sqrt(var_fg + eps)

        # === 3. 匹配背景统计 ===
        with torch.no_grad():
            mu_t, sigma_t = self.match_bg_stats(feat, top_k=3, num_shot=num_shot)

        # === 4. 构造扰动 ===
        feat_norm = (feat - mu_fg[:, :, None, None]) / (sigma_fg[:, :, None, None] + eps)
        feat_perturb = feat_norm * sigma_t[:, :, None, None] + mu_t[:, :, None, None]
        alpha = getattr(self, "perturb_strength", 0.4)
        feat_mix = (1 - alpha) * feat + alpha * feat_perturb

        # === ✅ 只在背景区域应用扰动 ===
        feat_out = feat * mask_fg + feat_mix * mask_bg

        # === 5. 可视化 ===
        if save_vis and img is not None:
            img_resized = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
            for b in range(B):
                img_np = TF.to_pil_image(img_resized[b].detach().cpu().clamp(0, 1))
                mask_np = mask[b, 0].detach().cpu().numpy()
                feat_map = feat_out[b, 0].detach().cpu().numpy()

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1);
                plt.imshow(img_np);
                plt.axis('off');
                plt.title("Input")
                plt.subplot(1, 3, 2);
                plt.imshow(mask_np, cmap='gray');
                plt.axis('off');
                plt.title("Mask")
                plt.subplot(1, 3, 3);
                plt.imshow(feat_map, cmap='viridis');
                plt.axis('off');
                plt.title("Perturbed Feat")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}_vis_{b}.png"), bbox_inches='tight', pad_inches=0.05)
                plt.close()

        return feat_out

    def load_background_library(self, bg_path: str, feat_dim: int):
        """
        加载背景库特征并注册缓冲区
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # === 加载背景特征 [N, 2, C] ===
            bg_feats = np.load(bg_path)
            bg_feats = torch.from_numpy(bg_feats).float().to(device)
            bg_mu = bg_feats[:, 0, :]
            bg_sigma = bg_feats[:, 1, :]

            # === 维度检查与投影 ===
            if bg_mu.shape[1] != feat_dim:
                print(f"[WARN] bg_dim={bg_mu.shape[1]} ≠ model_dim={feat_dim}, applying linear projection...")
                self.bg_proj = torch.nn.Linear(bg_mu.shape[1], feat_dim, bias=False).to(device)
                bg_mu = self.bg_proj(bg_mu.to(device))
                bg_sigma = self.bg_proj(bg_sigma.to(device))
            else:
                self.bg_proj = torch.nn.Identity().to(device)

            # === 数值安全处理 ===
            bg_mu = torch.nan_to_num(bg_mu, nan=0.0, posinf=0.0, neginf=0.0)
            bg_sigma = torch.nan_to_num(bg_sigma, nan=1.0, posinf=1.0, neginf=1.0)
            bg_sigma = torch.clamp(bg_sigma, min=1e-4, max=2.0)

            # === 生成归一化嵌入 ===
            bg_embed = F.normalize(bg_mu, dim=1)

            # === 注册缓冲区 ===
            self.register_buffer('bg_mu', bg_mu)
            self.register_buffer('bg_sigma', bg_sigma)
            self.register_buffer('bg_embed', bg_embed)

            print(f"[INFO] ✅ Loaded background library: {bg_mu.shape[0]} samples, dim={feat_dim}")

        except FileNotFoundError:
            print(f"[WARN] ❌ Background library not found at {bg_path}. Using default values.")
            self.bg_proj = torch.nn.Identity().to(device)
            self.register_buffer('bg_mu', torch.zeros(1, feat_dim, device=device))
            self.register_buffer('bg_sigma', torch.ones(1, feat_dim, device=device))
            self.register_buffer('bg_embed', torch.zeros(1, feat_dim, device=device))

    @abstractmethod
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              'feat_pos', and other algorithm-specific arguments.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask', and
              other algorithm-specific arguments.
        """
        pass

    @abstractmethod
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        """Forward with Transformer encoder.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output and other algorithm-specific
            arguments.
        """
        pass

    @abstractmethod
    def pre_decoder(self, memory: Tensor, **kwargs) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and other algorithm-specific arguments.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """
        pass

    @abstractmethod
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output, `references` including
            the initial and intermediate reference_points, and other
            algorithm-specific arguments.
        """
        pass
