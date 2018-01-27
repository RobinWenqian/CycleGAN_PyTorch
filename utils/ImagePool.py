import torch
from torch.autograd import Variable
import random

class ImagePool:
    def __init__(self, pool_size = 50):
        self.pool_size = 50
        self.images = []
    
    
    def fetch_one(self, generated_image):
        if len(self.images) < self.pool_size:
            self.images.append(generated_image)
            to_return = generated_image
        
        else:
            whether_to_add = random.uniform(0, 1)
            if whether_to_add > 0.5:
                pos = random.randint(0, self.pool_size - 1)
                to_return = self.images[pos].clone()
                self.images[pos] = generated_image            
            else:
                to_return = generated_image
        
        return Variable(to_return)