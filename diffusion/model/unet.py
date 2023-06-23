from diffusion.model.unet_blocks import encoder, decoder, SinusoidalPositionEmbeddings
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_filters=32, n_channels=3, bilinear=False, k_size=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        layers = [n_filters * x for x in [2, 4, 8, 16]]
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.inc = nn.Conv2d(n_channels, n_filters, 3, padding=1)

        self.encoder1 = encoder(n_filters, layers[0], time_emb_dim)
        self.encoder2 = encoder(layers[0], layers[1], time_emb_dim)
        self.encoder3 = encoder(layers[1], layers[2], time_emb_dim)
        self.encoder4 = encoder(layers[2], layers[3], time_emb_dim)

        # self.bridge1 = nn.Conv2d(layers[3], layers[3], k_size, padding=1)
        # self.bridge2 = nn.Conv2d(layers[3], layers[3], k_size, padding=1)

        self.decoder1 = decoder(layers[3], layers[2], time_emb_dim, bilinear, k_size)
        self.decoder2 = decoder(layers[2], layers[1], time_emb_dim, bilinear, k_size)
        self.decoder3 = decoder(layers[1], layers[0], time_emb_dim, bilinear, k_size)
        self.decoder4 = decoder(layers[0], n_filters, time_emb_dim, bilinear, k_size)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        x1 = self.inc(x)

        x2 = self.encoder1(x1, t) # 32 64
        x3 = self.encoder2(x2, t) # 64 128
        x4 = self.encoder3(x3, t) # 128 256
        x5 = self.encoder4(x4, t) # 256 512
        print(x5.shape)
        print("------------")

        # x = self.bridge1(x5)
        # x = self.bridge2(x)

        x = torch.cat((x, x5), dim=1)
        x = self.decoder1(x, t)
        # print(x.shape)
        # print(x5.shape)
        # print(x.shape)

        x = torch.cat((x, x4), dim=1)
        x = self.decoder2(x, t)

        x = self.decoder3(x, t)
        x = torch.cat((x, x3), dim=1)

        x = self.decoder4(x, t)
        x = torch.cat((x, x2), dim=1)

        # x = self.decoder1(x, t)
        # x = self.decoder2(x, t)
        # x = self.decoder3(x, t)
        # x = self.decoder4(x, t)

        out = self.outc(x)
        return out

