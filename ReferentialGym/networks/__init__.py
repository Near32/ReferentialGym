from .networks import FCBody, LSTMBody, GRUBody 
from .networks import ConvolutionalBody, ConvolutionalLstmBody, ConvolutionalGruBody 
from .networks import ModelResNet18, MHDPA_RN
from .networks import ConvolutionalMHDPABody, ResNet18MHDPA
from .networks import layer_init, hasnan, handle_nan

from .autoregressive_networks import ResNetBetaVAE

from .homoscedastic_multitask_loss import HomoscedasticMultiTasksLoss 

import torch.nn.functional as F 

def choose_architecture( architecture, 
                         kwargs=None,
                         hidden_units_list=None,
                         input_shape=None,
                         feature_dim=None, 
                         nbr_channels_list=None, 
                         kernels=None, 
                         strides=None, 
                         paddings=None,
                         dropout=0.0,
                         MHDPANbrHead=4,
                         MHDPANbrRecUpdate=1,
                         MHDPANbrMLPUnit=512,
                         MHDPAInteractionDim=128):
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

    if architecture == 'CNN-MHDPA':
        channels = [input_shape[0]] + nbr_channels_list
        body = ConvolutionalMHDPABody(input_shape=input_shape,
                                      feature_dim=feature_dim,
                                      channels=channels,
                                      kernel_sizes=kernels,
                                      strides=strides,
                                      paddings=paddings,
                                      dropout=dropout,
                                      nbrHead=MHDPANbrHead,
                                      nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                                      units_per_MLP_layer=MHDPANbrMLPUnit,
                                      interaction_dim=MHDPAInteractionDim)

    if 'ResNet18' in architecture and not("MHDPA" in architecture):
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        body = ModelResNet18(input_shape=input_shape,
                             feature_dim=feature_dim,
                             nbr_layer=nbr_layer,
                             pretrained=pretrained)
    elif 'ResNet18' in architecture and "MHDPA" in architecture:
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        body = ResNet18MHDPA(input_shape=input_shape,
                             feature_dim=feature_dim,
                             nbr_layer=nbr_layer,
                             pretrained=pretrained,
                             dropout=dropout,
                             nbrHead=MHDPANbrHead,
                             nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                             units_per_MLP_layer=MHDPANbrMLPUnit,
                             interaction_dim=MHDPAInteractionDim)

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
    if 'BetaVAE' in architecture:
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        beta = kwargs['vae_beta']
        maxCap = kwargs['vae_max_capacity']
        nbrEpochTillMaxEncodingCapacity = kwargs['vae_nbr_epoch_till_max_capacity']
        body = ResNetBetaVAE(beta=beta,
                             input_shape=input_shape,
                             latent_dim=feature_dim,
                             resnet_nbr_layer=nbr_layer,
                             pretrained=pretrained,
                             maxEncodingCapacity=maxCap,
                             nbrEpochTillMaxEncodingCapacity=nbrEpochTillMaxEncodingCapacity)
    
    return body