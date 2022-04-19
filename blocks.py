from cv2 import transform
import torch
from torch import nn

from layers import EqConv2d, EqLinear, Scaler, latent_to_style, AdaIn, NoiseLayer
from utils import equalizer
        
    
class InputBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super(InputBlock, self).__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.const_weight = nn.Parameter(torch.randn(1, in_feature, 4, 4))
        self.style1 = latent_to_style(dlatent_size, in_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = equalizer(Scaler(in_feature))
        self.noise2 = equalizer(Scaler(out_feature))
        #self.noise1 = NoiseLayer(in_feature)
        #self.noise2 = NoiseLayer(out_feature)
        self.adaIn1 = AdaIn(in_feature)
        self.adaIn2 = AdaIn(out_feature)
        self.Conv2d = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, latent, noise):
        output = self.noise1(self.const_weight)
        output = self.adaIn1(output, self.style1(latent))
        output = self.Active(self.Conv2d(output))
        output = self.noise2(output)
        output = self.adaIn2(output, self.style2(latent))   
        return output     

class SynBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dlatent_size = 512):
        super(SynBlock, self).__init__()
        self.num_feature = in_feature
        self.out_feature = out_feature
        self.Conv1 = EqConv2d(in_feature, out_feature, 3, 1, 1)
        self.Conv2 = EqConv2d(out_feature, out_feature, 3, 1, 1)
        self.style1 = latent_to_style(dlatent_size, out_feature)
        self.style2 = latent_to_style(dlatent_size, out_feature)
        self.noise1 = equalizer(Scaler(out_feature))
        self.noise2 = equalizer(Scaler(out_feature))
        self.adaIn = AdaIn(out_feature)
        self.Active = nn.LeakyReLU(0.2)
    
    def forward(self, x, latent, noise):
        output = output + self.noise1(noise)
        output = self.adaIn(output, self.style1(latent))
        output = self.Active(self.Conv2(output))
        output = output + self.noise2(noise)
        output = self.adaIn(output, self.style2(latent))
        output = self.Active(output)
        return output 
    
class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size_1, padding_1, 
                 kernel_size_2 = None, padding_2 = None):
        super(ConvBlock,self).__init__()
        if kernel_size_2 == None:
            kernel_size_2 = kernel_size_1
        if padding_2 == None:
            padding_2 = padding_1
        
        self.conv = nn.Sequential(
            EqConv2d(in_feature, out_feature, kernel_size_1, padding_1),
            nn.LeakyReLU(0.2),
            EqConv2d(out_feature, out_feature, kernel_size_2, padding_2),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        output = self.conv(x)
        return output

  