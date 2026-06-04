"""Convolutional backbones for image-space EBiEOT costs and potentials."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaNet(nn.Module):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x).squeeze()


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels: int, sub_sample: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


class NonlocalNet(nn.Module):
    def __init__(self, n_c: int = 3, n_f: int = 32, leak: float = 0.05):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),
            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),
            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.convs(x).view(x.shape[0], -1)
        return self.fc(conv_out).squeeze()
