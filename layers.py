import torch
from torch import nn
from utils import equalizer

class EqLinear(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        layer = nn.Linear(in_ch, out_ch)
        layer.weight.data.normal_()
        layer.bias.data.zero_()
        self.layer = equalizer(layer)
    
    def forward(self, x):
        return self.layer(x)

class EqConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride = 1, padding = 0, bias = True):
        super(EqConv2d, self).__init__()
        layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        layer.weight.data.normal_()
        layer.bias.data.zero_()
        self.layer = equalizer(layer)
    
    def forward(self, x):
        return self.layer(x)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
    
class Scaler(nn.Module):
    def __init__(self, num_feature):
        super(Scaler, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_feature, 1, 1))
        
    def forward(self, input):
        return input * self.weight
    
class latent_to_style(nn.Module):
    def __init__(self, dlatent_size = 512, style_ch = 512):
        super(latent_to_style, self).__init__()
        self.dlatent_size = dlatent_size
        self.style_ch = style_ch
        self.transformation = EqLinear(dlatent_size, style_ch * 2)
        self.transform.linear.bias.data[:style_ch] = 1
        self.transform.linear.bias.data[style_ch:] = 0
    
    def forward(self, latent):
        return self.transformation(latent).unsqueeze(2).unsqueeze(3)

class AdaIn(nn.Module):
    def __init__(self, style_ch):
        super(AdaIn, self).__init__()
        self.layer = nn.InstanceNorm2d(style_ch)
    def forward(self, input, style):
        weight, bias = style.chunk(2, 1)
        output = self.layer(input) * weight + bias
        return output

class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super(NoiseLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None
    def forward(self, x, noise = None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device = x.device, dtype = x.dtype)
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x
