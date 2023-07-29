import numpy as np
import torch
import random
import torchvision.transforms.functional as F
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

def random_transformation():
    p = random.random()
    if p < 0.2:
        return lambda x : x
    elif p<0.3:
        return lambda x : F.rotate(x, 90)
    elif p<0.4:
        return lambda x : F.rotate(x, 180)
    elif p<0.5:
        return lambda x : F.rotate(x, 270)
    elif p<0.75: 
        return lambda x : torch.flip(x, (1,))
    else: 
        return lambda x : torch.flip(x, (2,))
    
class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            if self.max_key != 'max':
                maximum = -1
            else:
                maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        f = random_transformation()
        input = f(input.unsqueeze(0)).squeeze(0)
        target = f(target.unsqueeze(0)).squeeze(0)
        return input, target, maximum, fname, slice
