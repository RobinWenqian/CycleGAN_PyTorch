from .Generator import Generator
from .Discriminator import Discriminator
from .loss import *
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import itertools
import sys
sys.path.append('..')
from utils.ImagePool import ImagePool
from torch.optim import lr_scheduler

class CycleGAN(nn.Module):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.initialize(opt)
    
    def deal_with_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A = input_A.cuda()
        self.input_B = input_B.cuda()
        
    def init_weights(self, m):
    #currently using Kaiming Initialization.
        if type(m) == nn.Linear:
            init.normal(m.weight.data, 0, 0.02)
        if type(m) == nn.Conv2d:
            init.normal(m.weight.data, 0, 0.02)
        if type(m) == nn.BatchNorm2d:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0)
            
    def initialize(self, opt):
        self.generated_As = ImagePool(opt.pool_size)
        self.generated_Bs = ImagePool(opt.pool_size)
        
        g_num_residual_blocks = opt.g_num_residual_blocks
        g_input_channels = opt.g_input_channels
        g_output_channels = opt.g_output_channels
        g_residual_block_channels = opt.g_residual_block_channels
        g_padding_type = opt.g_padding_type
        
        
        d_input_channels = opt.d_input_channels
        d_output_channels = opt.d_output_channels
        d_kernel_size = opt.d_kernel_size
        d_padding = opt.d_padding
        d_internal_layers = opt.d_internal_layers
        
        self.G_AB = Generator(g_num_residual_blocks, g_input_channels, g_output_channels,
                              g_residual_block_channels, g_padding_type)
        self.G_BA = Generator(g_num_residual_blocks, g_input_channels, g_output_channels,
                              g_residual_block_channels, g_padding_type)
        self.D_A = Discriminator(d_input_channels, d_output_channels, d_kernel_size,
                                 d_padding, d_internal_layers)
        self.D_B = Discriminator(d_input_channels, d_output_channels, d_kernel_size,
                                 d_padding, d_internal_layers)
        
        self.G_AB.apply(self.init_weights)
        self.G_BA.apply(self.init_weights)
        self.D_A.apply(self.init_weights)
        self.D_B.apply(self.init_weights)
        
        self.CycleLoss = nn.L1Loss()
        self.GANLoss = GANLoss()
        
        if opt.istrain:
            self.lambda_cycle = opt.lambda_cycle
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), 
                                                lr = opt.lr, betas = (opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

            self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

            #Adjusting the learning rate according to the paper.
            #First 100 epoches, lr fixed(0.0002). Next 100 epoches, lr will linearly decay to 0.
            milestone = opt.epoches // 2
            def schedule(epoch):
                if epoch < milestone:
                    return 1
                else:
                    return 1 - (epoch - milestone + 1)/ milestone

            self.lr_schedulers = [lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda = schedule),
                                  lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda = schedule),
                                  lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda = schedule)]
        
    def forward(self):
        self.input_A = Variable(self.input_A)
        self.input_B = Variable(self.input_B)

    def test(self):
        self.input_A = Variable(self.input_A, volatile = True)
        self.input_B = Variable(self.input_B, volatile = True)
        generated_A = self.G_BA(self.input_B)
        generated_B = self.G_AB(self.input_A)
        cycled_A = self.G_BA(generated_B)
        cycled_B = self.G_AB(generated_A)
        self.generated_A = generated_A.data[0]
        self.generated_B = generated_B.data[0]
        self.cycled_A = cycled_A.data[0]
        self.cycled_B = cycled_B.data[0]
        
    #self.input_A and self.input_B are Variables.
    def loss_G_backward(self):
        generated_A = self.G_BA(self.input_B)
        generated_B = self.G_AB(self.input_A)
        cycled_A = self.G_BA(generated_B)
        cycled_B = self.G_AB(generated_A)
        cycleloss_AA = self.lambda_cycle * self.CycleLoss(cycled_A, self.input_A)
        cycleloss_BB = self.lambda_cycle * self.CycleLoss(cycled_B, self.input_B)
        loss_G_AB = self.GANLoss(self.D_B(generated_B), True)
        loss_G_BA = self.GANLoss(self.D_A(generated_A), True)
        loss_G = cycleloss_AA + cycleloss_BB + loss_G_AB + loss_G_BA
        
        #backup the data
        self.cycleloss_AA = cycleloss_AA.data[0]
        self.cycleloss_BB = cycleloss_BB.data[0]
        self.loss_G_AB = loss_G_AB.data[0]
        self.loss_G_BA = loss_G_BA.data[0]
        self.loss_G = loss_G.data[0]
        self.generated_A = generated_A.data
        self.generated_B = generated_B.data
        self.cycled_A = cycled_A.data
        self.cycled_B = cycled_B.data
        loss_G.backward()
    
    #generated_A and generated_B are fetched from a pool of 50 history images. (Section 4 of Zhu et al's paper)
    def loss_D_backward(self):
        pred_input_A = self.D_A(self.input_A)
        pred_input_B = self.D_B(self.input_B)
        generated_A = self.generated_As.fetch_one(self.generated_A)
        generated_B = self.generated_Bs.fetch_one(self.generated_B)
        #[?]detach
        pred_generated_A = self.D_A(generated_A.detach())
        pred_generated_B = self.D_B(generated_B.detach())
        loss_D_A = 0.5 * self.GANLoss(pred_input_A, True) + 0.5 * self.GANLoss(pred_generated_A, False)
        loss_D_B = 0.5 * self.GANLoss(pred_input_B, True) + 0.5 * self.GANLoss(pred_generated_B, False)
        loss_D = loss_D_A + loss_D_B
        self.loss_D_A = loss_D_A.data[0]
        self.loss_D_B = loss_D_B.data[0]
        self.loss_D = loss_D.data[0]
    
    def loss_D_A_backward(self):
        pred_input_A = self.D_A(self.input_A)
        generated_A = self.generated_As.fetch_one(self.generated_A)
        pred_generated_A = self.D_A(generated_A.detach())
        loss_D_A = 0.5 * self.GANLoss(pred_input_A, True) + 0.5 * self.GANLoss(pred_generated_A, False)
        self.loss_D_A = loss_D_A.data[0]
        loss_D_A.backward()

    def loss_D_B_backward(self):
        pred_input_B = self.D_B(self.input_B)
        generated_B = self.generated_Bs.fetch_one(self.generated_B)
        pred_generated_B = self.D_B(generated_B.detach())
        loss_D_B = 0.5 * self.GANLoss(pred_input_B, True) + 0.5 * self.GANLoss(pred_generated_B, False)
        self.loss_D_B = loss_D_B.data[0]
        loss_D_B.backward()
        
    def optimize(self):
        self.forward()
        #optimizing G
        #Note: G should be optimized first, since the generated_A and generated_B needed when updating Ds
        #won't be available until G is optimized.
        self.optimizer_G.zero_grad()
        self.loss_G_backward()
        self.optimizer_G.step()
        #optimizing D_A
        self.optimizer_D_A.zero_grad()
        self.loss_D_A_backward()
        self.optimizer_D_A.step()
        #optimizing D_B
        self.optimizer_D_B.zero_grad()
        self.loss_D_B_backward()
        self.optimizer_D_B.step()
    
    def adjust_learning_rate(self):
        for scheduler in self.lr_schedulers:
            scheduler.step()
        print('Current learning rate is: %.6f' %(self.optimizer_G.param_groups[0]['lr']))