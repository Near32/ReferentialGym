from .networks import FCBody, LSTMBody, GRUBody 
from .networks import ConvolutionalBody, ConvolutionalLstmBody, ConvolutionalGruBody 
from .networks import ModelResNet18, MHDPA_RN
from .networks import ConvolutionalMHDPABody, ResNet18MHDPA
from .networks import layer_init, hasnan, handle_nan

from .autoregressive_networks import BetaVAE, MONet, ParallelMONet

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
        nbr_layer = None
        resnet_encoder = ('ResNet' in architecture)
        if resnet_encoder:
            nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        beta = kwargs['vae_beta']
        factor_vae_gamma = 0.0
        if 'factor_vae_gamma' in kwargs:
            factor_vae_gamma = kwargs['factor_vae_gamma']
        maxCap = kwargs['vae_max_capacity']
        nbrEpochTillMaxEncodingCapacity = kwargs['vae_nbr_epoch_till_max_capacity']
        nbr_attention_slot = None
        if 'vae_nbr_attention_slot' in kwargs:
            nbr_attention_slot = kwargs['vae_nbr_attention_slot']
        latent_dim = feature_dim
        if 'vae_nbr_latent_dim' in kwargs:
            latent_dim = kwargs['vae_nbr_latent_dim']
        decoder_nbr_layer = 4
        if 'vae_decoder_nbr_layer' in kwargs:
            decoder_nbr_layer = kwargs['vae_decoder_nbr_layer']
        if 'vae_decoder_conv_dim' in kwargs:
            decoder_conv_dim = kwargs['vae_decoder_conv_dim']
        NormalOutputDistribution = True
        if 'vae_use_gaussian_observation_model' in kwargs:
            NormalOutputDistribution = kwargs['vae_use_gaussian_observation_model']
        
        body = BetaVAE(beta=beta,
                       input_shape=input_shape,
                       latent_dim=latent_dim,
                       nbr_attention_slot=nbr_attention_slot,
                       resnet_encoder=resnet_encoder,
                       resnet_nbr_layer=nbr_layer,
                       pretrained=pretrained,
                       decoder_nbr_layer=decoder_nbr_layer,
                       decoder_conv_dim=decoder_conv_dim,
                       NormalOutputDistribution=NormalOutputDistribution,
                       maxEncodingCapacity=maxCap,
                       nbrEpochTillMaxEncodingCapacity=nbrEpochTillMaxEncodingCapacity,
                       factor_vae_gamma=factor_vae_gamma)

    if 'MONet' in architecture:
        beta = kwargs['vae_beta']
        gamma = kwargs['monet_gamma']

        resnet_encoder = ('ResNet' in architecture)
        nbr_layer = 2
        if resnet_encoder:
            nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        
        constrainedEncoding = False
        if 'vae_constrainedEncoding' in kwargs:
            constrainedEncoding = kwargs['vae_constrainedEncoding'] 
        if constrainedEncoding:
            maxCap = kwargs['vae_max_capacity']
            nbrEpochTillMaxEncodingCapacity = kwargs['vae_nbr_epoch_till_max_capacity']
        else:
            maxCap = 1.0
            nbrEpochTillMaxEncodingCapacity = 1

        nbr_attention_slot = 10
        if 'monet_nbr_attention_slot' in kwargs:
            nbr_attention_slot = kwargs['monet_nbr_attention_slot']
        latent_dim = feature_dim
        if 'vae_nbr_latent_dim' in kwargs:
            latent_dim = kwargs['vae_nbr_latent_dim']
        
        decoder_nbr_layer = 4
        if 'vae_decoder_nbr_layer' in kwargs:
            decoder_nbr_layer = kwargs['vae_decoder_nbr_layer']
        decoder_conv_dim =32
        if 'vae_decoder_conv_dim' in kwargs:
            decoder_conv_dim = kwargs['vae_decoder_conv_dim']
        
        anet_block_depth = 3 
        if 'monet_anet_block_depth' in kwargs:
            anet_block_depth = kwargs['monet_anet_block_depth']

        observation_sigma = 0.05
        if 'vae_observation_sigma' in kwargs:
            observation_sigma = kwargs['vae_observation_sigma']

        compactness_factor = None
        if 'unsup_seg_factor' in kwargs:
            compactness_factor = kwargs['unsup_seg_factor']

        arch = MONet 
        if 'Parallel' in architecture:
            arch = ParallelMONet
        
        body = arch(gamma=gamma,
                     input_shape=input_shape, 
                     nbr_attention_slot=nbr_attention_slot,
                     anet_basis_nbr_channel=32,
                     anet_block_depth=anet_block_depth,
                     cvae_beta=beta, 
                     cvae_latent_dim=latent_dim,
                     cvae_decoder_conv_dim=decoder_conv_dim, 
                     cvae_pretrained=pretrained, 
                     cvae_resnet_encoder=resnet_encoder,
                     cvae_resnet_nbr_layer=nbr_layer,
                     cvae_decoder_nbr_layer=decoder_nbr_layer,
                     cvae_maxEncodingCapacity=maxCap,
                     cvae_nbrEpochTillMaxEncodingCapacity=nbrEpochTillMaxEncodingCapacity,
                     cvae_constrainedEncoding=constrainedEncoding,
                     cvae_observation_sigma=observation_sigma,
                     compactness_factor=compactness_factor)

    return body