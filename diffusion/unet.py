from diffusion.unet_blocks import XConv, encoder, decoder, SinusoidalPositionEmbeddings

import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_filters=32, n_channels=3, bilinear=False, k_size=3, conv_per_layer=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        layers = [n_filters * x for x in [2, 4, 8, 16]]
        time_dim = 32

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.ReLU()
            )

        self.inc = nn.Conv2d(n_channels, n_filters, 3, padding=1)

        self.encoder1 = encoder(n_filters, layers[0])
        self.encoder2 = encoder(layers[0], layers[1])
        self.encoder3 = encoder(layers[1], layers[2])
        self.encoder4 = encoder(layers[2], layers[3])

        self.bridge1 = XConv(layers[3], layers[3], k_size)
        self.bridge2 = XConv(layers[3], layers[3], k_size)

        self.decoder1 = decoder(layers[3], layers[2], bilinear, k_size)
        self.decoder2 = decoder(layers[2], layers[1], bilinear, k_size)
        self.decoder3 = decoder(layers[1], layers[0], bilinear, k_size)
        self.decoder4 = decoder(layers[0], n_filters, bilinear, k_size)

        self.outc = nn.Conv2d(n_filters, n_channels, 1)

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



# class UNet(nn.Module):
#     def __init__(self, n_filters=32, n_channels=3, bilinear=False, k_size=3, conv_per_layer=2):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels

#         layers = [n_filters * x for x in [2, 4, 8, 16]]
#         time_dim = 32

#         self.time_mlp = nn.Sequential(
#                 SinusoidalPositionEmbeddings(time_dim),
#                 nn.Linear(time_dim, time_dim),
#                 nn.ReLU()
#             )

#         self.inc = (XConv(n_channels, n_filters))

#         self.encoder1 = encoder(n_filters, layers[0])
#         self.encoder2 = encoder(layers[0], layers[1])
#         self.encoder3 = encoder(layers[1], layers[2])
#         self.encoder4 = encoder(layers[2], layers[3])

#         self.bridge1 = XConv(layers[3], layers[3], k_size, conv_per_layer)
#         self.bridge2 = XConv(layers[3], layers[3], k_size, conv_per_layer)

#         self.decoder1 = decoder(layers[3], layers[2], bilinear, k_size)
#         self.decoder2 = decoder(layers[2], layers[1], bilinear, k_size)
#         self.decoder3 = decoder(layers[1], layers[0], bilinear, k_size)
#         self.decoder4 = decoder(layers[0], n_filters, bilinear, k_size)

#         self.outc = OutConv(n_filters, n_channels)

#     def forward(self, x, timestep):
#         t = self.time_mlp(timestep)

#         x1 = self.inc(x, t)

#         x2 = self.encoder1(x1, t)
#         x3 = self.encoder2(x2, t)
#         x4 = self.encoder3(x3, t)
#         x5 = self.encoder4(x4, t)

#         x = self.bridge1(x5, t)
#         x = self.bridge2(x, t)

#         x = self.decoder1(x, x4, t)
#         x = self.decoder2(x, x3, t)
#         x = self.decoder3(x, x2, t)
#         x = self.decoder4(x, x1, t)

#         out = self.outc(x)
#         return out
