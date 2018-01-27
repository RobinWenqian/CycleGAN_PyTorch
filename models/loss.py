import torch
import torch.nn as nn
from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        #We use mean square error loss as is pointed out in Section 4 of Zhu et al's paper.
        self.loss = nn.MSELoss()
    
    def get_target_tensor(self, input, target):
        if target:
            target_tensor = torch.ones(input.size())
        else:
            target_tensor = torch.zeros(input.size())
        target_tensor_var = Variable(target_tensor, requires_grad = False)
        return target_tensor_var
    
    def __call__(self, input, target):
        target_tensor_var = self.get_target_tensor(input, target).cuda()
        return self.loss(input, target_tensor_var)
