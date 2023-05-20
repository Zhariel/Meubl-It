import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class XConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, nb_conv=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            *[nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=1),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)
              ] * nb_conv
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            XConv(in_channels, out_channels, k_size=k_size)
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, k_size=3):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = XConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = XConv(in_ch, out_ch, k_size=k_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
