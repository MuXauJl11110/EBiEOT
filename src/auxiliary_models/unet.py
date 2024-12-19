import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=4, base_filters=64):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_channels (int): Number of output channels (e.g., 3 for RGB).
            num_layers (int): Number of encoder/decoder layers.
            base_filters (int): Number of filters in the first layer. Subsequent layers double this.
        """
        super(UNet, self).__init__()

        self.num_layers = num_layers

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        filters = base_filters
        for i in range(num_layers):
            self.encoders.append(self.conv_block(in_channels if i == 0 else filters // 2, filters))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            filters *= 2

        # Bottleneck
        self.bottleneck = self.conv_block(filters // 2, filters)

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.ups.append(self.upconv(filters, filters // 2))
            self.decoders.append(self.conv_block(filters, filters // 2))
            filters //= 2

        # Output layer
        self.output_layer = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Creates a convolutional block with two Conv2D layers followed by ReLU activations.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        """
        Creates an up-convolution (transposed convolution) layer.
        """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder path
        enc_features = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i in range(self.num_layers):
            x = self.ups[i](x)
            skip_connection = enc_features[self.num_layers - 1 - i]
            x = torch.cat((x, skip_connection), dim=1)  # Concatenate along channel dimension
            x = self.decoders[i](x)

        # Output layer
        return self.output_layer(x)
