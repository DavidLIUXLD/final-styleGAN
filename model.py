import enum
from turtle import forward
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from blocks import InputBlock, SynBlock
from layers import EqLinear, EqConv2d

class mapping(nn.Module):
    
    def __init__(self, latent_size = 512, map_out_size = 512, dlatent_size = 512, layer_size = 8, dlatent_broadCast = False):
        super(mapping, self).__init__()
        self.latent_size = latent_size
        self.dlatent_size = dlatent_size
        self.map_out_size = map_out_size
        self.layer_size = layer_size
        self.dlatent_broadCast = dlatent_broadCast
        
        active_layer = nn.LeakyReLU(negative_slope=0.2)
        layers = []
        init_layer = EqLinear(latent_size, map_out_size)
        layers.append(init_layer)
        layers.append(active_layer)
        for i in range(1, layer_size):
            map_in = map_out_size
            map_out = dlatent_size if i == layer_size - 1 else map_out_size
            layer = EqLinear(map_in, map_out)
            layers.append(layer)
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer(x)
        #todo: broadcast
        return x

class sythesis(nn.Module):
    def __init__(self, dlatent_size = 512, in_ch = 512):
        super(sythesis, self).__init__()
        self.InBlock = InputBlock(in_ch, 512, dlatent_size)
        synBlocks = []
        rgbBlocks = []
        for i in range(6):
            in_n = 512
            if(i < 3):
                synBlocks.append(SynBlock(512, 512, dlatent_size))
            else:
                synBlocks.append(SynBlock(in_n, in_n // 2, dlatent_size))
                in_n = in_n // 2
            if(i == 6):
                rgbBlocks.append(EqConv2d(in_n // 2, 3, 1))
        self.synLayer = nn.ModuleList(synBlocks)
        self.rgbLayer = nn.Sequential(OrderedDict(rgbBlocks))
    
    def forward(self, latent):
        x = self.InBlock(latent)
        for i, blocks in enumerate(self.synLayer):
            x = blocks(x, latent)
        x = self.rgbLayer(x)
             
class generator(nn.Module):
    def __init__(self, latent_size = 512, dlatent_size = 512):
        super(generator, self).__init__()
        self.mapping = mapping(latent_size = latent_size, dlatent_size = dlatent_size)
        self.synthesis = sythesis(dlatent_size=dlatent_size, in_ch = 512)
        
    def forward(self, latent):
        dlatent = self.mapping(latent)
        output = self.synthesis(dlatent)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fromRGB = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.LeakyReLU(0.2)
            
        )
        ndf = 16
        self.layer = nn.Sequential(
            #512
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #256 
            nn.Conv2d(16, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #128
            nn.Conv2d(16, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #64
            nn.Conv2d(16, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.layer(x)
    
        
    