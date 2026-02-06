"""
MRLSANet Model Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DPMultiScaleGating(nn.Module):
    """Dual-Path Multi-Scale Feature Gating Module"""

    def __init__(self, in_channels, pool_sizes=(1, 2, 4)):
        super().__init__()

        # Path A: standard 3x3 convolutions (Eq. 1)
        # FA = ReLU(BN(Conv3x3(ReLU(BN(Conv3x3(X))))))
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Path B: dilated convolutions (Eq. 2)
        # FB = ReLU(BN(Conv3x3_d4(ReLU(BN(Conv3x3_d2(ReLU(BN(Conv1x1(X)))))))))
        self.path_b = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Soft gating
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1),
        )

        # Forget gate
        self.forget_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Multi-scale pooling
        self.pool_sizes = pool_sizes
        self.pool_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            for _ in pool_sizes
        ])

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Dual paths
        a = self.path_a(x)
        b = self.path_b(x)

        # Gating weights
        g = self.gate_fc(x)
        g = F.softmax(g, dim=1)
        a_w, b_w = g[:, 0:1], g[:, 1:2]
        fused = a_w * a + b_w * b

        # Residual "forget" gate
        forget = torch.sigmoid(self.forget_conv(x))
        resid = x * forget
        out = fused + resid

        # Multi-scale pooled features
        multi = 0
        for ps, conv in zip(self.pool_sizes, self.pool_convs):
            p = F.adaptive_avg_pool2d(out, ps)
            p = conv(p)
            multi = multi + F.interpolate(
                p, size=out.shape[2:], mode='bilinear', align_corners=False
            )

        return self.final_relu(out + multi)


class ResidualLocalSelfAttention(nn.Module):
    """Residual Local Self-Attention Module"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = in_channels

        # Project to value
        self.value_conv = nn.Conv2d(in_channels, mid, 1, bias=False)

        # Local key/query (depth-wise 3x3)
        self.key_conv = nn.Conv2d(in_channels, mid, 3, padding=1, groups=in_channels, bias=False)
        self.query_conv = nn.Conv2d(in_channels, mid, 3, padding=1, groups=in_channels, bias=False)

        # Project back out
        self.out_conv = nn.Conv2d(mid, out_channels, 1, bias=False)

        # Residual path
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        v = self.value_conv(x)
        k = self.key_conv(x)
        q = self.query_conv(x)

        attn = torch.sigmoid(k * q)
        attended = attn * v
        out = self.out_conv(attended)
        res = self.res_conv(x)

        return self.relu(self.bn(out + res))


class MRLSANet(nn.Module):
    """MRLSA-Net: Multi-scale Residual Local Self-Attention Network"""

    def __init__(
        self,
        n_seg_classes=1,
        n_cls_classes=None,
        in_channels=3,
        backbone="timm-efficientnet-b5"
    ):
        super().__init__()

        # Encoder selection
        backbone = backbone.lower()
        if backbone == "timm-efficientnet-b5":
            weights = "noisy-student"
        elif backbone == "resnet50":
            weights = "imagenet"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.encoder = smp.encoders.get_encoder(
            backbone, in_channels=in_channels, depth=5, weights=weights
        )
        enc_chs = self.encoder.out_channels

        # Segmentation branch
        self.dpmfg = DPMultiScaleGating(enc_chs[-1])
        self.dec4 = ResidualLocalSelfAttention(enc_chs[-1] + enc_chs[-2], enc_chs[-2])
        self.dec3 = ResidualLocalSelfAttention(enc_chs[-2] + enc_chs[-3], enc_chs[-3])
        self.dec2 = ResidualLocalSelfAttention(enc_chs[-3] + enc_chs[-4], enc_chs[-4])
        self.dec1 = ResidualLocalSelfAttention(enc_chs[-4] + enc_chs[-5], enc_chs[-5])
        self.seg_head = nn.Conv2d(enc_chs[-5], n_seg_classes, kernel_size=1)

        # Classification branch (optional)
        if n_cls_classes is not None and n_cls_classes > 0:
            self.cls_pool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(enc_chs[-1], enc_chs[-1] // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(enc_chs[-1] // 2, n_cls_classes)
            )
        else:
            self.cls_head = None

    def forward(self, x, mode="seg"):
        """
        Forward pass
        mode: "seg" (segmentation only), "cls" (classification only), "all" (both)
        """
        feats = self.encoder(x)
        x5 = feats[-1]

        # Classification path
        cls_logits = None
        if mode in ("cls", "all") and self.cls_head is not None:
            c = self.cls_pool(x5)
            cls_logits = self.cls_head(c)

        # Segmentation path
        seg_map = None
        if mode in ("seg", "all"):
            out = self.dpmfg(x5)

            out = F.interpolate(out, size=feats[-2].shape[2:], mode='bilinear', align_corners=False)
            out = self.dec4(torch.cat([out, feats[-2]], dim=1))

            out = F.interpolate(out, size=feats[-3].shape[2:], mode='bilinear', align_corners=False)
            out = self.dec3(torch.cat([out, feats[-3]], dim=1))

            out = F.interpolate(out, size=feats[-4].shape[2:], mode='bilinear', align_corners=False)
            out = self.dec2(torch.cat([out, feats[-4]], dim=1))

            out = F.interpolate(out, size=feats[-5].shape[2:], mode='bilinear', align_corners=False)
            out = self.dec1(torch.cat([out, feats[-5]], dim=1))

            out = self.seg_head(out)
            seg_map = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        if mode == "all":
            return seg_map, cls_logits
        elif mode == "seg":
            return seg_map
        elif mode == "cls":
            return cls_logits
        else:
            raise ValueError(f"Invalid mode: {mode}")


def get_model(config):
    """Helper function to create MRLSANet model"""
    return MRLSANet(
        n_seg_classes=config.NUM_CLASSES,
        n_cls_classes=None,
        in_channels=config.IN_CHANNELS,
        backbone=config.BACKBONE
    )