from cv2 import transform
import torch
import math
from torch import nn
from layers import EqConv2d, EqLinear, Scaler, latent_to_style, AdaIn, NoiseLayer
        
    
class InputBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super(InputBlock, self).__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.const_weight = nn.Parameter(torch.randn(1, in_feature, 4, 4))
        self.style1 = latent_to_style(dlatent_size, in_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = NoiseLayer(in_feature)
        self.noise2 = NoiseLayer(out_feature)
        self.adaIn1 = AdaIn(in_feature)
        self.adaIn2 = AdaIn(out_feature)
        self.Conv2d = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, latent):
        output = self.noise1(self.const_weight)
        output = self.adaIn1(output, self.style1(latent))
        output = self.Active(self.Conv2d(output))
        output = self.noise2(output)
        output = self.adaIn2(output, self.style2(latent))        

class SynBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super(SynBlock, self).__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Conv1 = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Conv2 = EqConv2d(out_feature, out_feature, 3, 1, 1)
        self.style1 = latent_to_style(dlatent_size, out_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = NoiseLayer(out_feature)
        self.noise2 = NoiseLayer(out_feature)
        self.adaIn = AdaIn(out_feature)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, x, latent):
        output = self.upSample(x)
        output = self.Active(self.Conv1(output))
        output = self.noise1(output)
        output = self.adaIn(output, self.style1(latent))
        output = self.Active(self.Conv2(output))
        output = self.noise2(output)
        output = self.adaIn(output, self.style2(latent)) 

  