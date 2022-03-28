import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from blocks import InputBlock, SynBlock
from layers import EqLinear, EqConv2d

class mapping(nn.Module):
    
    def __init__(self, latent_size = 512, map_out_size = 512, dlatent_size = 512, layer_size = 8, dlatent_broadCast = False):
        super.__init__()
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
        self.layer = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.layer(x)
        #todo: broadcast
        return x

class sythesis(nn.Module):
    def __init__(self, dlatent_size = 512, in_ch = 512, out_ch = 512):
        self.InBlock = InputBlock(in_ch, out_ch, dlatent_size)
        synBlocks = []
        for i in range(6):
            in_n = in_ch
            if (i < 3):
            
        
        
    