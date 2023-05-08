from diffusion.unet_blocks import XConv, Encoder, Decoder, OutConv, SinusoidalPositionEmbeddings

import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_filters=32, n_channels=3, bilinear=False, k_size=3, conv_per_layer=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        l = [n_filters * x for x in [2, 4, 8, 16]]
        time_dim = 32

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.ReLU()
            )

        self.inc = (XConv(n_channels, n_filters))

        self.encoder1 = Encoder(n_filters, l[0])
        self.encoder2 = Encoder(l[0], l[1])
        self.encoder3 = Encoder(l[1], l[2])
        self.encoder4 = Encoder(l[2], l[3])

        self.bridge1 = XConv(l[3], l[3], k_size, conv_per_layer)
        self.bridge2 = XConv(l[3], l[3], k_size, conv_per_layer)

        self.decoder1 = Decoder(l[3], l[2], bilinear, k_size)
        self.decoder2 = Decoder(l[2], l[1], bilinear, k_size)
        self.decoder3 = Decoder(l[1], l[0], bilinear, k_size)
        self.decoder4 = Decoder(l[0], n_filters, bilinear, k_size)

        self.outc = OutConv(n_filters, n_channels)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        x1 = self.inc(x, t)

        x2 = self.encoder1(x1, t)
        x3 = self.encoder2(x2, t)
        x4 = self.encoder3(x3, t)
        x5 = self.encoder4(x4, t)

        x = self.bridge1(x5, t)
        x = self.bridge2(x, t)

        x = self.decoder1(x, x4, t)
        x = self.decoder2(x, x3, t)
        x = self.decoder3(x, x2, t)
        x = self.decoder4(x, x1, t)

        out = self.outc(x)
        return out
