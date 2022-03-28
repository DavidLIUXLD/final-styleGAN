from cv2 import transform
import torch
import math
from torch import nn
from layers import EqConv2d, EqLinear, Scaler

class latent_to_style(nn.Module):
    def __init__(self, dlatent_size = 512, style_ch = 512):
        super.__init__()
        self.dlatent_size = dlatent_size
        self.style_ch = style_ch
        self.transformation = EqLinear(dlatent_size, style_ch)
        self.transformation.layer.weight[:style_ch] = 1
        self.transformation.layer.bias[style_ch:] = 0
    
    def forward(self, latent):
        return self.transformation(latent).unsqueeze(2).unsqueeze(3)

class AdaIn(nn.Module):
    def __init__(self, style_ch):
        super.__init__()
        self.layer = nn.InstanceNorm2d(style_ch)
    def forward(self, input, style):
        weight, bias = style.chunk(2,1)
        output = self.layer(input) * weight + bias
        return output        
    
class InputBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super.__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.const_weight = nn.Parameter(torch.randn(1, in_feature, 4, 4))
        self.style1 = latent_to_style(dlatent_size, in_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = Scaler(in_feature)
        self.noise2 = Scaler(out_feature)
        self.adaIn1 = AdaIn(in_feature)
        self.adaIn2 = AdaIn(out_feature)
        self.Conv2d = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, latent, noise):
        output = self.const_weight + self.noise1(noise)
        output = self.adaIn1(output, self.style1(latent))
        output = self.Active(self.Conv2d(output))
        output = output + self.noise2(noise)
        output = self.adaIn2(output, self.style2(latent))        

class SynBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super.__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Conv1 = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Conv2 = EqConv2d(out_feature, out_feature, 3, 1, 1)
        self.style1 = latent_to_style(dlatent_size, out_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = Scaler(out_feature)
        self.noise2 = Scaler(out_feature)
        self.adaIn = AdaIn(out_feature)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, latent, noise, input):
        output = self.upSample(input)
        output = self.Active(self.Conv1(output))
        output = self.adaIn(output + self.noise1(noise), latent)
        output = self.Active(self.Conv2(output))
        output = self.adaIn(output + self.noise2(noise), latent) 

  