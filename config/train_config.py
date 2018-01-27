import argparse
import os


class TrainConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--name', required = True, help = 'Please specify a model name.')
        self.parser.add_argument('--data_dir', required = True, help = 'The folder where training/test set is stored.')
        self.parser.add_argument('--log_dir', type = str, default = 'log/txt', help = 
                                 'The folder where loggings are stored.')
        self.parser.add_argument('--snapshot_dir', type = str, default = 'log/snapshot', help = 
                                 'The folder where snapshots are stored.')
        self.parser.add_argument('--istrain', type = bool, default = True, help = 'Whether it is training mode.')
        self.parser.add_argument('--shuffle', type = bool, default = True, help = 'Whether to shuffle the data.')
        self.parser.add_argument('--epoches', type = int, default = 200, help = 'Epoches.')
        self.parser.add_argument('--g_num_residual_blocks', type = int, default = 9, help = 
                                 'Residual blocks used in generator.')
        self.parser.add_argument('--g_input_channels', type = int, default = 3, help = 
                                 'Input channels for generator.')
        self.parser.add_argument('--g_output_channels', type = int, default = 3, help = 
                                 'Output channels for generator.')
        self.parser.add_argument('--g_residual_block_channels', type = int, default = 128, help = 
                                 'Residual block channels for generator.')
        self.parser.add_argument('--g_padding_type', type = str, default = 'reflection', help = 
                                 'Padding type for generator. Valid input: reflection, replication and zero.')
        self.parser.add_argument('--d_input_channels', type = int, default = 3, help = 
                                 'Input channels for discriminator.')
        self.parser.add_argument('--d_output_channels', type = int, default = 64, help = 
                                 'Output channels for discriminator. (Note: output refers to the output of 1st conv. layer.)')
        self.parser.add_argument('--d_kernel_size', type = int, default = 4, help = 
                                 'Kernel size used in discriminator.')
        self.parser.add_argument('--d_padding', type = int, default = 1, help = 
                                 'Padding size used in discriminator.')
        self.parser.add_argument('--d_internal_layers', type = int, default = 3, help = 
                                 'The number of stride 2 conv. layers used in discriminator.')
        self.parser.add_argument('--lambda_cycle', type = int, default = 10, help = 
                                 'The coefficient of cyclic loss mentioned in the paper.')
        self.parser.add_argument('--pool_size', type = int, default = 50, help = 
                                 'Size of Image Pool used to update the discriminator.')
        self.parser.add_argument('--lr', type = float, default = 0.0002, help = 'Learning rate.')
        self.parser.add_argument('--beta1', type = float, default = 0.5, help = 'Momentum for Adam optimizer.')
        self.initialized = True
    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt