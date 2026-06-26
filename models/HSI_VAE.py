import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def apply_layer_norm(
        x: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.apply_layer_norm(x, self.norm1)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.apply_layer_norm(x, self.norm2)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class HSIEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        base_channels: int = 64,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.level1 = nn.Sequential(
            *[
                ResBlock(base_channels)
                for _ in range(num_res_blocks)
            ]
        )

        self.down1 = Downsample(
            base_channels,
            base_channels * 2,
        )

        self.level2 = nn.Sequential(
            *[
                ResBlock(base_channels * 2)
                for _ in range(num_res_blocks)
            ]
        )

        self.down2 = Downsample(
            base_channels * 2,
            base_channels * 4,
        )

        self.bottleneck = nn.Sequential(
            *[
                ResBlock(base_channels * 4)
                for _ in range(num_res_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(base_channels * 4)

        self.output_conv = nn.Conv2d(
            base_channels * 4,
            latent_channels * 2,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)

        x = self.level1(x)
        x = self.down1(x)

        x = self.level2(x)
        x = self.down2(x)

        x = self.bottleneck(x)

        x = x.permute(0, 2, 3, 1)
        x = self.output_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = F.silu(x)
        x = self.output_conv(x)

        mu, logvar = torch.chunk(x, chunks=2, dim=1)

        logvar = torch.clamp(
            logvar,
            min=-20.0,
            max=10.0,
        )

        return mu, logvar


class HSIDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 31,
        base_channels: int = 64,
        latent_channels: int = 16,
        num_res_blocks: int = 4,
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            latent_channels,
            base_channels * 4,
            kernel_size=3,
            padding=1,
        )

        self.bottleneck = nn.Sequential(
            *[
                ResBlock(base_channels * 4)
                for _ in range(num_res_blocks)
            ]
        )

        self.up1 = Upsample(
            base_channels * 4,
            base_channels * 2,
        )

        self.level2 = nn.Sequential(
            *[
                ResBlock(base_channels * 2)
                for _ in range(num_res_blocks)
            ]
        )

        self.up2 = Upsample(
            base_channels * 2,
            base_channels,
        )

        self.level1 = nn.Sequential(
            *[
                ResBlock(base_channels)
                for _ in range(num_res_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(base_channels)

        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(z)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.level2(x)

        x = self.up2(x)
        x = self.level1(x)

        x = x.permute(0, 2, 3, 1)
        x = self.output_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = F.silu(x)
        x = self.output_conv(x)

        return x


class HSIVAE(nn.Module):
    def __init__(
        self,
        hsi_channels: int = 31,
        base_channels: int = 64,
        latent_channels: int = 16,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.encoder = HSIEncoder(
            in_channels=hsi_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
        )

        self.decoder = HSIDecoder(
            out_channels=hsi_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)

        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        z, mu, logvar = self.encode(x, sample=sample)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, z
