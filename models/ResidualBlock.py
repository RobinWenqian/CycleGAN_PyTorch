import torch
import torch.nn as nn

def add_padding(padding_type, sequence, pad_size = 1):
    if padding_type == 'reflection':
        sequence += [nn.ReflectionPad2d(pad_size)]
    elif padding_type == 'replication':
        sequence += [nn.ReplicationPad2d(pad_size)]
    elif padding_type == 'zero':
        return 
    else:
        assert 0 == 1, print(padding_type, 'is invalid!')
    return 0

def add_conv(sequence, kernel_size, input_channels, output_channels, stride, padding):
    sequence += [nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, 
                           stride = stride, padding = padding, bias = True),
                 nn.InstanceNorm2d(output_channels),
                 nn.ReLU(True)]

class ResidualBlock(nn.Module):
    def __init__(self, channels = 128, padding_type = 'reflection'):
        super(ResidualBlock, self).__init__()
        sequence = []
        padding = add_padding(padding_type, sequence)      
        add_conv(sequence, 3, channels, channels, 1, padding)
        '''
        sequence += [nn.Dropout(0.5)]
        '''
        padding = add_padding(padding_type, sequence)
        sequence += [nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, bias = True)]
        #add_conv(sequence, 3, channels, channels, 1, padding)
        self.model = nn.Sequential(*sequence)
    
    
    def forward(self, data):
        return self.model(data) + data
        