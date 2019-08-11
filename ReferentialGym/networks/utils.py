import torch
import torchvision.transforms as T
import numpy as np


class ResizeNormalize(object):
    def __init__(self, size, use_cuda=False, normalize_rgb_values=False, toPIL=False):
        '''
        Used to resize, normalize and convert raw pixel observations.
        
        :param x: Numpy array to be processed
        :param size: int or tuple, (height,width) size
        :param use_cuda: Boolean to determine whether to create Cuda Tensor
        :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                     to interval (0-1)
        '''
        if isinstance(size, int): size = (size,size)
        ts = []
        if toPIL: ts.append(T.ToPILImage())
        ts.append(T.Resize(size=size))
        ts.append(T.ToTensor())
        
        self.scaling_operation = T.Compose(ts)
        self.normalize_rgb_values = normalize_rgb_values
        self.use_cuda = use_cuda

    def __call__(self, x):
        x = self.scaling_operation(x)
        # WATCHOUT: it is necessary to cast the tensor into float before doing
        # the division, otherwise the result is yielded as a uint8 (full of zeros...)
        x = x.type(torch.FloatTensor)
        x = x / 255. if self.normalize_rgb_values else x
        if self.use_cuda:
            return x.cuda()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'