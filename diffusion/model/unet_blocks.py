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


def encoder(in_ch, out_ch, time_emb_dim, k_size=3):
    conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    transpose_layer = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
    return XConv(in_ch, out_ch, time_emb_dim, conv1, transpose_layer, k_size=k_size)


def decoder(in_ch, out_ch, time_emb_dim, bilinear=False, k_size=3):
    conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    transpose_layer = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    return XConv(in_ch, out_ch, time_emb_dim, conv1, transpose_layer, k_size=k_size)


class XConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, conv1, transpose_layer, k_size=3):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = conv1
        self.transform = transpose_layer
        self.conv2 = nn.Conv2d(out_ch, out_ch, k_size, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)
