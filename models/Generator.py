import torch
import torch.nn as nn
import numpy as np
from .ResidualBlock import *


def add_deconv(sequence, kernel_size, input_channels, output_channels, stride, padding, output_padding):
    sequence += [nn.ConvTranspose2d(input_channels, output_channels, kernel_size = kernel_size,
                                    stride = stride, padding = padding, output_padding = output_padding, bias = True),
                 nn.InstanceNorm2d(output_channels),
                 nn.ReLU(True)]

def c7s1k(sequence, input_channels = 3, k = 32, padding_type = 'reflection'):
    padding = add_padding(padding_type, sequence, pad_size = 3)
    add_conv(sequence, 7, input_channels, k, 1, padding)

def downsampling(sequence, input_channels, output_channels, padding_type = 'reflection'):
    padding = add_padding(padding_type, sequence)
    add_conv(sequence, 3, input_channels, output_channels, 2, padding)

def upsampling(sequence, input_channels, output_channels):
    add_deconv(sequence, 3, input_channels, output_channels, 2, 1, 1)

def generator(num_residual_blocks, input_channels = 3, output_channels = 3, residual_block_channels = 128, 
              padding_type = 'reflection'):
    '''
    The structure of generator is c7s1-32,d64,d128,serveral 128-128 residual blocks, u64, u32, c7s1-3.
    The code c7s1-k, dk, uk are defined in the paper.
    '''
    rbc = residual_block_channels
    #rbc: residual_block_channels for short
    sequence = []
    c7s1k(sequence, input_channels = input_channels, k = rbc//4, padding_type = padding_type)
    downsampling(sequence, rbc//4, rbc//2, padding_type = padding_type)
    downsampling(sequence, rbc//2, rbc, padding_type = padding_type)
    for i in range(num_residual_blocks):
        sequence += [ResidualBlock(channels = rbc, padding_type = padding_type)]
    upsampling(sequence, rbc, rbc//2)
    upsampling(sequence, rbc//2, rbc//4)
    
    add_padding(padding_type, sequence, pad_size = 3)
    #NOTE: DON'T add ReLU before Tanh (restricting output)
    sequence += [nn.Conv2d(rbc//4, 3, kernel_size = 7, stride = 1, bias = True),
                 nn.InstanceNorm2d(3)]
    #c7s1k(sequence, input_channels = rbc//4, k = 3, padding_type = padding_type)
    sequence += [nn.Tanh()]
    return sequence


class Generator(nn.Module):
    def __init__(self, num_residual_blocks, input_channels = 3, output_channels = 3, residual_block_channels = 128, 
             padding_type = 'reflection'):
        super(Generator, self).__init__()
        sequence = generator(num_residual_blocks, input_channels, output_channels,
                             residual_block_channels, padding_type)
        self.model = nn.Sequential(*sequence)
    
    
    def forward(self, data):
        return self.model(data)