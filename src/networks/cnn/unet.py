import torch
import torch.nn as nn


class UNet2(nn.Module):
    """U-Net used as a generative cost map ``x -> y`` for colored MNIST."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_layers: int = 4,
        base_filters: int = 64,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        filters = base_filters
        for i in range(num_layers):
            self.encoders.append(
                self._conv_block(in_channels if i == 0 else filters // 2, filters)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            filters *= 2

        self.bottleneck = self._conv_block(filters // 2, filters)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.ups.append(self._upconv(filters, filters // 2))
            self.decoders.append(self._conv_block(filters, filters // 2))
            filters //= 2

        self.output_layer = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _upconv(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        for i in range(self.num_layers):
            x = self.ups[i](x)
            skip = enc_features[self.num_layers - 1 - i]
            x = torch.cat((x, skip), dim=1)
            x = self.decoders[i](x)

        return self.output_layer(x)
