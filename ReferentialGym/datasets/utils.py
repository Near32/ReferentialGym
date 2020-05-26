import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image 


class DictBatch(object):
    def __init__(self, data):
        """
        :param data: list of Dict of Tensors.
        """
        self.keys = list(data[0].keys())
        values = list(zip(*[list(d.values()) for d in data]))

        for idx, key in enumerate(self.keys):
            setattr(self, key, torch.cat(values[idx], dim=0))
        
    def pin_memory(self):
        for key in self.keys:
            attr = getattr(self, key).pin_memory()
            setattr(self, key, attr)
        return self

    def cuda(self):
        for key in self.keys:
            attr = getattr(self, key).cuda()
            setattr(self, key, attr)
        return self

    def keys(self):
        return self.keys

    def __getitem__(self, key):
        """
        :param key: str
        """
        return getattr(self, key, None)

def collate_dict_wrapper(batch):
    return DictBatch(batch)


class ResizeNormalize(object):
    def __init__(self, size, use_cuda=False, normalize_rgb_values=False, toPIL=False, rgb_scaler=1.0):
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
        self.rgb_scaler = rgb_scaler
        self.use_cuda = use_cuda

    def __call__(self, x):
        x = self.scaling_operation(x)
        # WATCHOUT: it is necessary to cast the tensor into float before doing
        # the division, otherwise the result is yielded as a uint8 (full of zeros...)
        x = x.type(torch.FloatTensor)
        x = x / 255. if self.normalize_rgb_values else x
        x *= self.rgb_scaler
        if self.use_cuda:
            return x.cuda()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AddEgocentricInvariance(object):
    def __init__(self, marker_demisize=2):
        '''
            Add a central marker to enable egocentric invariance.
            
            :param marker_demisize: Int, half the size of the marker.
        '''
        self.marker_demisize = marker_demisize
    
    def __call__(self, x):
        x = np.array(x)
        dim = x.shape[-2]
        marker_colour = x.max()
        start = int(dim//2-self.marker_demisize)
        end = int(dim//2+self.marker_demisize)
        x[start:end, :, ...] = marker_colour
        x[:,start:end, ...] = marker_colour
        x = Image.fromarray(x.astype('uint8'))
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Rescale(object) :
  def __init__(self, output_size) :
    assert( isinstance(output_size, (int, tuple) ) )
    self.output_size = output_size

  def __call__(self, sample) :
    image = sample
    h,w = image.shape[:2]
    new_h, new_w = self.output_size
    img = cv2.resize(image, (new_h, new_w) )
    return img


class RescaleNormalize(object):
    def __init__(self, size, use_cuda=False, normalize_rgb_values=False):
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
        ts.append(Rescale(output_size=size))
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
