import torch
import torchvision.transforms as T
import numpy as np

def PreprocessFunction(x, use_cuda=False):
    if use_cuda:
        return torch.from_numpy(x).type(torch.cuda.FloatTensor)
    else:
        return torch.from_numpy(x).type(torch.FloatTensor)

def ResizeCNNPreprocessFunction(x, size, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to resize, normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param size: int or tuple, (height,width) size
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    if isinstance(size, int): size = (size,size)
    scaling_operation = T.Compose([T.ToPILImage(),
                                    T.Resize(size=size)])
    if len(x.shape) == 4:
        #batch:
        imgs = []
        for i in range(x.shape[0]):
            img = x[i]
            # handled Stacked images...
            per_image_first_channel_indices = range(0,img.shape[-1]+1,3)
            img = np.concatenate( [ np.array(scaling_operation(img[...,idx_begin:idx_end])) for idx_begin, idx_end in zip(per_image_first_channel_indices,per_image_first_channel_indices[1:])], axis=-1)
            img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
            imgs.append(img)
        x = torch.cat(imgs, dim=0)
    else:
        x = scaling_operation(x)
        x = np.array(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
    # WATCHOUT: it is necessary to cast the tensor into float before doing
    # the division, otherwise the result is yielded as a uint8 (full of zeros...)
    x = x.type(torch.FloatTensor) / 255. if normalize_rgb_values else x
    if use_cuda:
        return x.type(torch.cuda.FloatTensor)
    return x.type(torch.FloatTensor)


def CNNPreprocessFunction(x, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    x = x.transpose((2, 0, 1))
    x = x / 255. if normalize_rgb_values else x
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)
