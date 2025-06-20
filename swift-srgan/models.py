import torch
from torch import nn



class SeperableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        #super(SeperableConv2d, self).__init__()
        print(f"SeperableConv2d: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, bias={bias}")
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride = stride,
            groups=in_channels,
            bias=bias,
            padding=padding
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=1,
            bias=bias
        )
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class StackedSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=5, stride=1, padding=1, bias=True):
        super().__init__()
        layers = []
        for i in range(num_layers):
            input_c = in_channels if i == 0 else out_channels
            print("Called from StackedSeparableConv")
            layers.append(
                SeperableConv2d(input_c, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class StackedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, kernel_size=3, num_layers=5, use_bn=True, discriminator=False, **kwargs):
        #super(ConvBlock, self).__init__()
        super().__init__()
        
        self.use_act = use_act
        self.cnn = StackedSeparableConv(in_channels, out_channels, num_layers, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
    
class StackedFinalConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.block(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=True, discriminator=False, **kwargs):
        #super(ConvBlock, self).__init__()
        super().__init__()
        
        self.use_act = use_act
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        #super(UpsampleBlock, self).__init__()
        super().__init__()
        
        self.conv = SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor) # (in_channels * 4, H, W) -> (in_channels, H*2, W*2)
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x):
        return self.act(self.ps(self.conv(x)))
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        #super(ResidualBlock, self).__init__()
        super().__init__()
        
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
        
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x
    
    
class Generator(nn.Module):
    """Swift-SRGAN Generator
    Args:
        in_channels (int): number of input image channels.
        num_channels (int): number of hidden channels.
        num_blocks (int): number of residual blocks.
        upscale_factor (int): factor to upscale the image [2x, 4x, 8x].
    Returns:
        torch.Tensor: super resolution image
    """

    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        self.upscale_factor = upscale_factor
        
        #super(Generator, self).__init__()
        super().__init__()
        
        #self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.initial = StackedConvBlock(in_channels, num_channels, kernel_size=3, num_layers=5, stride=1, padding=4, use_bn=False)
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsampler = nn.Sequential(
            *[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)]
        )
        #self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        self.final_conv = StackedFinalConvBlock(num_channels, num_channels // 2, in_channels)
        
    def forward(self, x):

        # Bicubic or bilinear upsampling as global skip
        skip = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x) + initial
        x = self.upsampler(x)
        sr = self.final_conv(x)

        # Combine with skip
        sr = sr + skip  # Global skip connection

        return (torch.tanh(sr) + 1) / 2


class Discriminator(nn.Module):
    """Swift-SRGAN Discriminator
    Args:
        in_channels (int): number of input image channels.
        features (tuple): sequence of hidden channels.
    Returns:
        torch.Tensor
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: tuple = (64, 64, 128, 128, 256, 256, 512, 512),
    ) -> None: # Here "-> None" indicates return type which is optional
        #super(Discriminator, self).__init__()
        super().__init__()

        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2, # Controls the step size of the convolution kernel as it slides across the input(Kernel moves 1 pixel at a time (output size ≈ input size).
                    padding=1, # Adds zeros around the input to preserve spatial dimensions ().
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # Pool to fixed 6x6 size
            nn.Flatten(), # Flatten the output to 1D
            nn.Linear(512 * 6 * 6, 1024), # Dense layer, feature reduction
            nn.LeakyReLU(0.2, inplace=True), # Adds non-linearity
            nn.Linear(1024, 1), # Output a single score (real/fake probability input)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return torch.sigmoid(self.classifier(x))


