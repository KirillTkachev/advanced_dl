import torch.nn as nn
import torch


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):

    def __init__(self, input_channels = 6):
        super().__init__()
        self.model = nn.Sequential(
            Down(input_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.model(x)
        return x
