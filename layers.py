import torch
from torch import nn
from utils import equalizer

class EqLinear(nn.Module):
    def __init__(self, in_ch, out_ch):
        layer = nn.Linear(in_ch, out_ch)
        layer.weight.data.normal_()
        layer.bias.data.zero_()
        self.layer = equalizer(layer)
    
    def forward(self, x):
        return self.layer(x)

class EqConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride = 1, padding = 0, bias = True):
        layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias)
        layer.weight.data.normal_()
        layer.bias.data.zero_()
        self.layer = equalizer(layer)
    
    def forward(self, x):
        return self.layer(x)

class Scaler(nn.module):
    def __init__(self, num_feature):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_feature, 1, 1))
        
    def forwward(self, input):
        return input * self.weight