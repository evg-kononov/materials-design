import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Optional, List


class EqualizedWeight(nn.Module):
    
    def __init__(self, shape: List[int], lr_multiplier: float = 1.):
        """
        shape - the shape of the weight parameter
        lr_multiplier - learning rate multiplier
        """
        super().__init__()
        
        # He initialization constant
        self.c = (1 / math.sqrt(np.prod(shape[1:]))) * lr_multiplier
        # Initialize the weights with N(0,1)
        self.weight = nn.Parameter(torch.randn(shape) / lr_multiplier)
    
    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: float = 0., lr_multiplier: float = 1.):
        """
        in_features - the number of features in the input feature map
        out_features - is the number of features in the output feature map
        bias - the bias initialization constant
        lr_multiplier - learning rate multiplier
        """
        super().__init__()

        self.weight = EqualizedWeight([out_features, in_features], lr_multiplier)
        self.bias = nn.Parameter(torch.ones(out_features) * bias)
        self.lr_multiplier = lr_multiplier
    
    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias * self.lr_multiplier)


class EqualizedConv3d(nn.Module):
    
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        """
        in_features - the number of features in the input feature map
        out_features - the number of features in the output feature map
        kernel_size - the size of the convolution kernel
        padding - the padding to be added on both sides of each size dimension
        """
        super().__init__()
        
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features,
                                       kernel_size, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))
        
    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.weight(), bias=self.bias, padding=self.padding)


class Conv3dWeightModulate(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        """
        in_features - the number of features in the input feature map
        out_features - the number of features in the output feature map
        kernel_size - the size of the convolution kernel
        demodulate - flag whether to normalize weights by its standard deviation
        eps - the epsilon for normalizing
        """
        super().__init__()
        
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features,
                                       kernel_size, kernel_size, kernel_size])
        self.eps = eps
        
    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        x - the input feature map of shape [batch_size, in_features, height, width, deepth]
        s - style based scaling tensor of shape [batch_size, in_features]
        """
        
        b, _, h, w, d = x.shape
        # Reshape the scales
        s = s[:, None, :, None, None, None]
        # Get learning rate equalized weights
        weights = self.weight()[None, :, :, :, :, :]
        weights = weights * s
        
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * sigma_inv
        
        # Reshape x
        x = x.reshape(1, -1, h, w, d)
        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        # Use grouped convolution to efficiently calculate the
        # convolution with sample wise kernel, i.e., we have a
        # different kernel (weights) for each sample in the batch
        x = F.conv3d(x, weights, padding=self.padding, groups=b)
        # Reshape x to [batch_size, out_features, height, width, deepth] and return
        return x.reshape(-1, self.out_features, h, w, d)


class ToRGB(nn.Module):
    """
    Generates an RGB image from a feature map using 1x1x1 convolution.
    """
    
    def __init__(self, d_latent: int, features: int):
        """
        d_latent - the dimensionality of w
        features - the number of features in the feature map
        """
        super().__init__()
        
        # Get style vector from w
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        # Weight modulated convolution layer without demodulation
        self.conv = Conv3dWeightModulate(features, 1, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        x - the input feature map of shape [batch_size, in_features, height, width, deepth]
        w - w with shape [batch_size, d_latent]
        """
        
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None, None])


class Smooth(nn.Module):
    """
    This layer blurs each channel.
    """
    
    def __init__(self):
        super().__init__()
        
        # Gaussian blur
        l = 3
        simga = 1
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy, zz = np.meshgrid(ax, ax, ax)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * simga**2))
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad3d(1)
        
    def forward(self, x: torch.Tensor):
        b, c, h, w, d = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w, d)
        x = self.pad(x)
        x = F.conv3d(x, self.kernel)
        return x.view(b, c, h, w, d)


class UpSample(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.smooth = Smooth()
    
    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class DownSample(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.smooth = Smooth()
        
    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2),
                             mode='trilinear', align_corners=False)


class MappingNetwork(nn.Module):
    
    def __init__(self, features: int, n_layers: int, lr_multiplier: float = 0.01):
        """
        features - the number of features in z and w
        n_layers - the number of layers in the mapping network
        lr_multiplier - learning rate multiplier for the mapping layers
        """
        super().__init__()
        
        # Create the MLP
        layers = []
        for i in range(n_layers):
            # Equalized learning-rate linear layers
            layers.append(EqualizedLinear(features, features, lr_multiplier=lr_multiplier))
            # LeakyReLU
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor):
        # Normalize z
        z = F.normalize(z, dim=1)
        # Map z to w
        return self.net(z)


class MiniBatchStdDev(nn.Module):
    
    def __init__(self, group_size: int = 4):
        """
        group_size - the number of samples to calculate standard deviatinon
        """
        super().__init__()
        
        self.group_size = group_size
    
    def forward(self, x: torch.Tensor):
        """
        x - the feature map
        """
        
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        grouped = x.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1, 1)
        b, _, h, w, d = x.shape
        std = std.expand(b, -1, h, w, d)
        return torch.cat([x, std], dim=1)


class StyleBlock(nn.Module):
    
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        d_latent - the dimensionality of w
        in_features - the number of features in the input feature map
        out_features - the number of features in the output feature map
        """
        super().__init__()
        
        # Get style vector from w
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv3dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        x - the input feature map of shape [batch_size, in_features, height, width, deepth]
        w - w with shape [batch_size, d_latent]
        noise - a tensor of shape [batch_size, 1, height, width, deepth]
        """
        
        s = self.to_style(w)
        x = self.conv(x, s)
        
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None, None] * noise
        
        return self.activation(x + self.bias[None, :, None, None, None])


class GeneratorBlock(nn.Module):
    
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        d_latent - the dimensionality of w
        in_features - the number of features in the input feature map
        out_features - the number of features in the output feature map
        """
        super().__init__()
        
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_rgb = ToRGB(d_latent, out_features)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor,
                noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        x - the input feature map of shape [batch_size, in_features, height, width, deepth]
        w - w with shape [batch_size, d_latent]
        noise - a tuple of two noise tensors of shape [batch_size, 1, height, width, deepth]
        """
        
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb


class Generator(nn.Module):
    
    def __init__(self, log_resolution: int, d_latent: int,
                 n_features: int = 32, max_features: int = 512, activation=None):
        """
        log_resolution - the log2 of image resolution
        d_latent - the dimensionality of w
        n_features - number of features in the convolution layer at the highest resolution (final block)
        max_features - maximum number of features in any generator block
        """
        super().__init__()
        
        # Calculate the number of features for each block
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        # Number of generator blocks
        self.n_blocks = len(features)
        # Trainable 4x4x4 constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4, 4)))
        # First style block for 4x4x4 resolution and layer to get RGB
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])
        # Generator blocks
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        # 2x up sampling layer. The feature space is up sampled at each block
        self.up_sample = UpSample()
        # Weather to use output activation
        self.activation = activation
        
    def forward(self, w: List[Optional[torch.Tensor]],
                input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        w - w. In order to mix-styles (use different w for different layes),
        we provide a separate w for each generator block.
        It has shape [n_blocks, batch_size, d_latent].
        input_noise - the noise for each block. It's a list of pairs of noise sensors
        because each block (except the initial) has two noise inputs
        after each convolution layer.
        """
        
        batch_size = w.shape[1]
        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        
        # Evaluate rest of the blocks
        for i in range(1, self.n_blocks):
            # Up sample the feature map
            x = self.up_sample(x)
            # Run it through the generator block
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # Up sample the RGB image and add to the rgb from the block
            rgb = self.up_sample(rgb) + rgb_new
            
        if self.activation:
            return self.activation(rgb)
        else:
            return rgb


class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_features: int, out_features: int):
        """
        in_features - the number of features in the input feature map
        out_features - the number of features in the output feature map 
        """
        super().__init__()
        
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv3d(in_features, out_features, kernel_size=1))
        self.block = nn.Sequential(
            EqualizedConv3d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv3d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = DownSample()
        # Scalling factor after adding the residual
        self.scale = 1 / math.sqrt(2)
        
    def forward(self, x: torch.Tensor):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class Discriminator(nn.Module):
    
    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        log_resolution - the log2 of image resolution
        n_features - number of features in the convolution layer at the highest resolution (first block)
        max_features - maximum number of features in any generator block
        """
        super().__init__()
        
        # Layer to convert RGB image to a feature map
        self.from_rgb = nn.Sequential(
            EqualizedConv3d(in_features=1, out_features=n_features, kernel_size=1),
            nn.LeakyReLU(0.2, True),
        )
        # Calculate the number of features for each block
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # Number of discriminator blocks
        n_blocks = len(features) - 1
        # Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.std_dev = MiniBatchStdDev()
        # Number of features after adding the standard deviations map
        final_features = features[-1] + 1
        # Final 3x3x3 convolution layer
        self.conv = EqualizedConv3d(final_features, final_features, 3)
        # Final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * 2 * final_features, 1)
        
    def forward(self, x: torch.Tensor):
        """
        x - the input image of shape [batch_size, 1, height, width, deepth]
        """
        
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        return self.final(x)