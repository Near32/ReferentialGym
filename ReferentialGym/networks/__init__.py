from .networks import FCBody, LSTMBody, GRUBody, ConvolutionalBody, ConvolutionalLstmBody, ConvolutionalGruBody, ModelResNet18, layer_init, hasnan, handle_nan

import torch.nn.functional as F 

def choose_architecture( architecture, 
                         hidden_units_list=None,
                         input_shape=None,
                         feature_dim=None, 
                         nbr_channels_list=None, 
                         kernels=None, 
                         strides=None, 
                         paddings=None,
                         dropout=0.0):
    if 'LSTM-RNN' in architecture:
        return LSTMBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    
    if 'GRU-RNN' in architecture:
        return GRUBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    
    if architecture == 'MLP':
        return FCBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    
    if architecture == 'CNN':
        channels = [input_shape[0]] + nbr_channels_list
        body = ConvolutionalBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings,
                                     dropout=dropout)
    if 'ResNet18' in architecture:
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        body = ModelResNet18(input_shape=input_shape,
                             feature_dim=feature_dim,
                             nbr_layer=nbr_layer,
                             pretrained=pretrained)

    if architecture == 'CNN-RNN':
        channels = [input_shape[0]] + nbr_channels_list
        body = ConvolutionalLstmBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings,
                                     hidden_units=hidden_units_list,
                                     dropout=dropout)
    return body