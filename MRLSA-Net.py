import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from thop import profile, clever_format


class DPMultiScaleGating(nn.Module):
    def __init__(self, in_channels, mid_channels=None, pool_sizes=(1,2,4)):
        super().__init__()
        mid_channels = mid_channels or in_channels//2
        # Path A: standard 3×3
        self.path_a = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        # Path B: dilated
        self.path_b = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(midannels:=mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        # Soft gates
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, 2, 1),
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
        a = self.path_a(x)
        b = self.path_b(x)
        g = self.gate_fc(x)
        g = F.softmax(g, dim=1)
        a_w, b_w = g[:,0:1], g[:,1:2]
        fused = a_w*a + b_w*b

        forget = torch.sigmoid(self.forget_conv(x))
        resid = x * forget
        out = fused + resid

        multi = 0
        for ps, conv in zip(self.pool_sizes, self.pool_convs):
            p = F.adaptive_avg_pool2d(out, ps)
            p = conv(p)
            multi = multi + F.interpolate(p, size=out.shape[2:], mode='bilinear', align_corners=False)

        return self.final_relu(out + multi)


class ResidualLocalSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = in_channels // 2
        # project to “value”
        self.value_conv = nn.Conv2d(in_channels, mid, 1, bias=False)
        # local key / query both to mid channels
        self.key_conv   = nn.Conv2d(in_channels, mid, 3, padding=1, bias=False)
        self.query_conv = nn.Conv2d(in_channels, mid, 3, padding=1, bias=False)
        # project back out
        self.out_conv = nn.Conv2d(mid, out_channels, 1, bias=False)
        # residual path
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        v = self.value_conv(x)      # [B, mid, H, W]
        k = self.key_conv(x)        # [B, mid, H, W]
        q = self.query_conv(x)      # [B, mid, H, W]
        attn = torch.sigmoid(k * q) # [B, mid, H, W]
        attended = attn * v         # [B, mid, H, W]
        out = self.out_conv(attended)
        res = self.res_conv(x)
        return self.relu(self.bn(out + res))



class MRLSA-Net(nn.Module):
    def __init__(self, n_classes=1, in_channels=3, backbone="timm-efficientnet-b5"):
        super().__init__()
        # encoder
        self.encoder = smp.encoders.get_encoder(
            backbone, in_channels=in_channels, depth=5, weights="noisy-student"
        )
        enc_chs = self.encoder.out_channels  # e.g. [64,176,384,672,2048]
        # boundary refinement & enhancement
        self.dpmfg = DPMultiScaleGating(enc_chs[-1])
        # decoder stages (fuse with skips)
        self.dec4 = ResidualLocalSelfAttention(enc_chs[-1] + enc_chs[-2], enc_chs[-2])
        self.dec3 = ResidualLocalSelfAttention(enc_chs[-2] + enc_chs[-3], enc_chs[-3])
        self.dec2 = ResidualLocalSelfAttention(enc_chs[-3] + enc_chs[-4], enc_chs[-4])
        self.dec1 = ResidualLocalSelfAttention(enc_chs[-4] + enc_chs[-5], enc_chs[-5])
        # final seg head
        self.head = nn.Conv2d(enc_chs[-5], n_classes, 1)

    def forward(self, x):
        feats = self.encoder(x)
        x5 = feats[-1]
        x  = self.dpmfg(x5)
        # stage 4
        x = F.interpolate(x, size=feats[-2].shape[2:], mode='bilinear', align_corners=False)
        x = self.dec4(torch.cat([x, feats[-2]], dim=1))
        # stage 3
        x = F.interpolate(x, size=feats[-3].shape[2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, feats[-3]], dim=1))
        # stage 2
        x = F.interpolate(x, size=feats[-4].shape[2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, feats[-4]], dim=1))
        # stage 1
        x = F.interpolate(x, size=feats[-5].shape[2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, feats[-5]], dim=1))
        # head & upsample
        x = self.head(x)
        x = F.interpolate(x, size=(192,192), mode='bilinear', align_corners=False)
        return x

def get_model(n_classes=1):
    return MRLSA-Net(n_classes=n_classes)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(n_classes=3).to(device)
    print(model)

    # shape check
    x = torch.randn((1,3,192,192), device=device)
    y = model(x)
    print("Output:", y.shape)   # → [1,3,192,192]

    # FLOPs & params
    try:
        data = torch.randn((4,3,192,192), device=device)
        macs, params = profile(model, inputs=(data,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"FLOPs: {macs}")
        print(f"Parameters: {params}")
    except Exception as e:
        print("Error computing FLOPs/params:", e)
