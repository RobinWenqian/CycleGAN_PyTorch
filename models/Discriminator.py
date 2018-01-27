import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    '''
    We define a PatchGAN discriminator in this CycleGAN implementation.
    According to the paper, we use 3 conv. layers with (kernel size, stride, padding) = (4, 2, 1) and 2 conv. 
    layers with corresponding parameters (4, 1, 1).
    Thus, the size of perceptual field will be 70x70, which is pointed out in the paper.
    '''
    def __init__(self, input_channels = 3, output_channels = 64, kernel_size = 4, padding = 1, internal_layers = 3):
        super(Discriminator, self).__init__()
        sequence = [nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, stride = 2, padding = padding),
                    nn.LeakyReLU(0.2, True)]
        
        cur_output_channels = output_channels
        
        for i in range(internal_layers - 1):
            #The min function is aimed at preventing the feature maps from being too deep.
            new_output_channels = min(8*output_channels, cur_output_channels*2)
            sequence += [
                nn.Conv2d(cur_output_channels, new_output_channels, kernel_size = kernel_size, stride = 2, padding = padding),
                nn.InstanceNorm2d(new_output_channels),
                nn.LeakyReLU(0.2, True)]
            cur_output_channels = new_output_channels
        
        sequence += [
            nn.Conv2d(cur_output_channels, new_output_channels, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.InstanceNorm2d(new_output_channels),
            nn.LeakyReLU(0.2, True)]
        
        #Patch discriminator. output_(i,j) points out whether patch_(i,j) is real.
        sequence += [
            nn.Conv2d(new_output_channels, 1, kernel_size = kernel_size, stride = 1, padding = padding)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, data):
        return self.model(data)