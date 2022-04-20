import torch
from torch import nn
from blocks import ConvBlock, InputBlock, SynBlock
from layers import EqLinear, EqConv2d, PixelNorm

class mapping(nn.Module):
    
    def __init__(self, dlatent_size = 512, layer_size = 8, dlatent_broadCast = False):
        super(mapping, self).__init__()
        self.dlatent_size = dlatent_size
        self.layer_size = layer_size
        self.dlatent_broadCast = dlatent_broadCast
        
        active_layer = nn.LeakyReLU(negative_slope=0.2)
        layers = [PixelNorm()]
        for i in range(layer_size):
            layers.append(EqLinear(dlatent_size, dlatent_size))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer(x)
        return x

class sythesis(nn.Module):
    def __init__(self, dlatent_size = 512, in_ch = 512, n_syn = 7, im_ch = 512):
        super(sythesis, self).__init__()
        self.InBlock = InputBlock(in_ch, im_ch, dlatent_size)
        synBlocks = []
        in_n = im_ch
        for i in range(n_syn):
            if(i < n_syn // 2):
                synBlocks.append(SynBlock(in_ch, in_ch, dlatent_size))
            else:
                synBlocks.append(SynBlock(in_n, in_n // 2, dlatent_size))
                in_n = in_n // 2
        self.synLayer = nn.ModuleList(synBlocks)
        self.rgbprevLayer = EqConv2d(in_n*2, 3, 1)
        self.rgbLayer = nn.Sequential(EqConv2d(in_n, 3, 1), nn.LeakyReLU(0.2))
    
    def forward(self, latent, noise, alpha = -1):
        x = self.InBlock(latent, noise[0])
        x_prev = None
        for i, blocks in enumerate(self.synLayer):
            x_prev = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = blocks(x_prev, latent, noise[i+1])
        x = self.rgbLayer(x)
        if 0 <= alpha < 1:
            x_prev = self.rgbprevLayer(x_prev)
            x = alpha * x + (1 - alpha) * x_prev
        return x
         
class Generator(nn.Module):
    def __init__(self, latent_size = 512, dlatent_size = 512, n_syn = 7, im_ch = 512):
        super(Generator, self).__init__()
        self.mapping = mapping(dlatent_size)
        self.synthesis = sythesis(dlatent_size, latent_size, n_syn, im_ch)
        
    def forward(self, latent, noise, alpha = -1):
        dlatent = self.mapping(latent)
        output = self.synthesis(dlatent, noise, alpha)
        return output

class Discriminator(nn.Module):
    def __init__(self, n_cov = 8, out_ch = 512):
        super(Discriminator, self).__init__()
        self.n_cov = n_cov
        self.fromRGB = nn.ModuleList()
        self.layers = nn.ModuleList()
        ch = out_ch // (2 ** (n_cov // 2 + 1))
        for i in range(n_cov):
            self.fromRGB.append(EqConv2d(3, ch, 1))
            if i <= n_cov // 2:
                self.layers.append(ConvBlock(ch, ch * 2, 3, 1))
                ch = ch * 2
            else:
                if(i == n_cov - 1):
                    self.layers.append(ConvBlock(ch + 1, ch, 3, 1, 4, 0))
                else:
                    self.layers.append(ConvBlock(ch, ch, 3, 1))
        self.transformation = EqLinear(out_ch, 1)
    
    def forward(self, x, alpha = -1):
        output = None
        for i in range(self.n_cov):
            if i == 0:
                output = self.fromRGB[i](x)
            if i == self.n_cov - 1:
                output_var = output.var(0, False) + 1e-8
                output_std = torch.sqrt(output_var)
                mean_std = output_std.mean().expand(output.size(0), 1, 4, 4)
                output = torch.cat([output, mean_std], 1)
            output = self.layers[i](output)
            if i < self.n_cov - 1:
                output = nn.functional.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)
                if 0 <= alpha < 1 and i == 0:
                    output_next = self.fromRGB[i + 1](x)
                    output_next = nn.functional.interpolate(output_next, scale_factor=0.5, mode = 'bilinear', align_corners=False)
                    output = alpha * output + (1 - alpha) * output
                    
        output = output.squeeze(2).squeeze(2)
        output = self.transformation(output)
        return output