from .networks import FCBody, LSTMBody, GRUBody, MHDPA_RN 
from .networks import ConvolutionalBody, EntityPrioredConvolutionalBody, ConvolutionalLstmBody, ConvolutionalGruBody, ConvolutionalMHDPABody
from .residual_networks import ModelResNet18, ModelResNet18AvgPooled, ResNet18MHDPA, ResNet18AvgPooledMHDPA, ExtractorResNet18
from .networks import ModelVGG16, ExtractorVGG16

from .networks import layer_init, hasnan, handle_nan

from .autoregressive_networks import DeconvolutionalBody
from .autoregressive_networks import ResNetEncoder, ResNetAvgPooledEncoder, BroadcastingDecoder, ResNetParallelAttentionEncoder, ParallelAttentionBroadcastingDeconvDecoder
from .autoregressive_networks import BetaVAE, MONet, ParallelMONet

from .homoscedastic_multitask_loss import HomoscedasticMultiTasksLoss 

import torch.nn as nn 
import torch.nn.functional as F 

def choose_architecture( architecture, 
                         kwargs=None,
                         fc_hidden_units_list=None,
                         rnn_hidden_units_list=None,
                         input_shape=None,
                         output_shape=None,
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
        return LSTMBody(input_shape[0], hidden_units=rnn_hidden_units_list, gate=nn.LeakyReLU)
    
    if 'GRU-RNN' in architecture:
        return GRUBody(input_shape[0], hidden_units=rnn_hidden_units_list, gate=nn.LeakyReLU)
    
    if 'MLP' in architecture:
        return FCBody(input_shape, hidden_units=fc_hidden_units_list, non_linearities=[nn.LeakyReLU])
    
    if 'CNN' in architecture and 'DCNN' not in architecture:
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
            
        channels = [input_shape[0]] + nbr_channels_list
        if 'MHDPA' in architecture:
            body = ConvolutionalMHDPABody(input_shape=input_shape,
                                          feature_dim=feature_dim,
                                          channels=channels,
                                          kernel_sizes=kernels,
                                          strides=strides,
                                          paddings=paddings,
                                          fc_hidden_units=fc_hidden_units_list,
                                          dropout=dropout,
                                          nbrHead=MHDPANbrHead,
                                          nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                                          units_per_MLP_layer=MHDPANbrMLPUnit,
                                          interaction_dim=MHDPAInteractionDim)
        else:
            if 'EntityPriored' in architecture:
                body = EntityPrioredConvolutionalBody(input_shape=input_shape,
                                                     feature_dim=feature_dim,
                                                     channels=channels,
                                                     kernel_sizes=kernels,
                                                     strides=strides,
                                                     paddings=paddings,
                                                     fc_hidden_units=fc_hidden_units_list,
                                                     use_coordconv=use_coordconv,
                                                     dropout=dropout)
            else:
                body = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=feature_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings,
                                         fc_hidden_units=fc_hidden_units_list,
                                         use_coordconv=use_coordconv,
                                         dropout=dropout)
    elif 'DCNN' in architecture:
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
            
        channels = [input_shape[0]] + nbr_channels_list
        if 'MHDPA' in architecture:
            raise NotImplementedError
            """
            body = ConvolutionalMHDPABody(input_shape=input_shape,
                                          feature_dim=feature_dim,
                                          channels=channels,
                                          kernel_sizes=kernels,
                                          strides=strides,
                                          paddings=paddings,
                                          fc_hidden_units=fc_hidden_units_list,
                                          dropout=dropout,
                                          nbrHead=MHDPANbrHead,
                                          nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                                          units_per_MLP_layer=MHDPANbrMLPUnit,
                                          interaction_dim=MHDPAInteractionDim)
            """
        else:
            body = DeconvolutionalBody(input_shape=input_shape,
                                       output_shape=output_shape,
                                       channels=channels,
                                       kernel_sizes=kernels,
                                       strides=strides,
                                       paddings=paddings,
                                       use_coordconv=use_coordconv,
                                       dropout=dropout)

        
    if 'VGG16' in architecture:
        arch = architecture.copy()
        arch = arch.remove('VGG16').remove('-')
        pretrained = ('pretrained' in arch)
        if pretrained:  arch = arch.remove('pretrained')
        final_layer_idx = None 
        final_layer_idx = abs(int(arch))
        body = ModelVGG16(input_shape=input_shape,
                          feature_dim=feature_dim,
                          pretrained=pretrained,
                          final_layer_idx=final_layer_idx)

    if 'ResNet18AvgPooled' in architecture and not("MHDPA" in architecture):
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture.lower())
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
        body = ModelResNet18AvgPooled(input_shape=input_shape,
                                      feature_dim=feature_dim,
                                      nbr_layer=nbr_layer,
                                      pretrained=pretrained,
                                      use_coordconv=use_coordconv)
    elif 'ResNet18AvgPooled' in architecture and "MHDPA" in architecture:
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture.lower())
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
        body = ResNet18AvgPooledMHDPA(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     nbr_layer=nbr_layer,
                                     pretrained=pretrained,
                                     use_coordconv=use_coordconv,
                                     dropout=dropout,
                                     nbrHead=MHDPANbrHead,
                                     nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                                     units_per_MLP_layer=MHDPANbrMLPUnit,
                                     interaction_dim=MHDPAInteractionDim)

    elif 'ResNet18' in architecture and not("MHDPA" in architecture):
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture.lower())
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
        body = ModelResNet18(input_shape=input_shape,
                             feature_dim=feature_dim,
                             nbr_layer=nbr_layer,
                             pretrained=pretrained,
                             use_coordconv=use_coordconv)
        
    elif 'ResNet18' in architecture and "MHDPA" in architecture:
        nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture.lower())
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
        body = ResNet18MHDPA(input_shape=input_shape,
                             feature_dim=feature_dim,
                             nbr_layer=nbr_layer,
                             pretrained=pretrained,
                             use_coordconv=use_coordconv,
                             dropout=dropout,
                             nbrHead=MHDPANbrHead,
                             nbrRecurrentSharedLayers=MHDPANbrRecUpdate,
                             units_per_MLP_layer=MHDPANbrMLPUnit,
                             interaction_dim=MHDPAInteractionDim)

    elif architecture == 'CNN-RNN':
        channels = [input_shape[0]] + nbr_channels_list
        body = ConvolutionalLstmBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings,
                                     fc_hidden_units=fc_hidden_units_list,
                                     rnn_hidden_units=rnn_hidden_units_list,
                                     dropout=dropout)
    if 'BetaVAE' in architecture:
        use_coordconv = None
        if 'coord2' in architecture.lower():
            use_coordconv = 2 
        if 'coord4' in architecture.lower():
            use_coordconv = 4 
        resnet_encoder = ('ResNet' in architecture)
        resnet_nbr_layer = None
        use_avg_pooled = ('AvgPooled' in architecture)
        if resnet_encoder:
            resnet_nbr_layer = int(architecture[-1])
        pretrained = ('pretrained' in architecture)
        beta = kwargs['vae_beta']
        factor_vae_gamma = 0.0
        if 'factor_vae_gamma' in kwargs:
            factor_vae_gamma = kwargs['factor_vae_gamma']
        
        constrainedEncoding = False
        if 'vae_constrainedEncoding' in kwargs:
            constrainedEncoding = kwargs['vae_constrainedEncoding'] 
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
        

        if nbr_attention_slot is None:
            if resnet_encoder:
                if use_avg_pooled:
                    encoder = ResNetAvgPooledEncoder(input_shape=input_shape, 
                                            latent_dim=latent_dim,
                                            nbr_layer=resnet_nbr_layer,
                                            pretrained=pretrained,
                                            use_coordconv=use_coordconv)
                else:
                    encoder = ResNetEncoder(input_shape=input_shape, 
                                            latent_dim=latent_dim,
                                            nbr_layer=resnet_nbr_layer,
                                            pretrained=pretrained,
                                            use_coordconv=use_coordconv)
            else:
                channels = [input_shape[0]] + nbr_channels_list
                encoder = ConvolutionalBody(input_shape=input_shape,
                                            feature_dim=[feature_dim, latent_dim*2],
                                            channels=channels,
                                            kernel_sizes=kernels,
                                            strides=strides,
                                            paddings=paddings,
                                            fc_hidden_units=fc_hidden_units_list,
                                            use_coordconv=use_coordconv,
                                            dropout=dropout)
                '''
                encoder = ConvolutionalBody(input_shape=input_shape,
                                            feature_dim=(256, latent_dim*2), 
                                            channels=[input_shape[0], 32, 32, 64], 
                                            kernel_sizes=[8, 4, 3],#[3, 3, 3], 
                                            strides=[2, 2, 2],
                                            paddings=[1, 1, 1],#[0, 0, 0],
                                            dropout=0.0,
                                            non_linearities=[F.relu],
                                            use_coordconv=use_coordconv)
                '''
                '''
                encoder = ConvolutionalMHDPABody(input_shape=input_shape,
                                                feature_dim=(256, latent_dim*2),
                                                channels=[input_shape[0], 32, 32, 64],
                                                kernel_sizes=[3, 3, 3],
                                                strides=[2, 2, 2],
                                                paddings=[0, 0, 0],
                                                dropout=0.0,
                                                nbrHead=4,
                                                nbrRecurrentSharedLayers=1,
                                                units_per_MLP_layer=256,
                                                interaction_dim=128,
                                                non_linearities=[F.relu],
                                                use_coordconv=use_coordconv)
                '''
            decoder = BroadcastingDecoder(output_shape=input_shape,
                                           net_depth=decoder_nbr_layer, 
                                           kernel_size=3, 
                                           stride=1, 
                                           padding=1, 
                                           latent_dim=latent_dim, 
                                           conv_dim=decoder_conv_dim)
            '''
            decoder = BroadcastingDeconvDecoder(output_shape=input_shape,
                                               net_depth=decoder_nbr_layer, 
                                               latent_dim=latent_dim, 
                                               conv_dim=decoder_conv_dim)
            '''
        else:
            encoder_latent_dim = latent_dim
            latent_dim *= nbr_attention_slot
            encoder = ResNetParallelAttentionEncoder(input_shape=input_shape, 
                                                     latent_dim=encoder_latent_dim,
                                                     nbr_attention_slot=nbr_attention_slot,
                                                     nbr_layer=resnet_nbr_layer,
                                                     pretrained=pretrained,
                                                     use_coordconv=use_coordconv)
            '''
            encoder = ResNetPHDPAEncoder(input_shape=input_shape, 
                                        latent_dim=encoder_latent_dim,
                                        nbr_attention_slot=nbr_attention_slot,
                                        nbr_layer=resnet_nbr_layer,
                                        pretrained=pretrained,
                                        use_coordconv=use_coordconv)
            '''
            decoder = ParallelAttentionBroadcastingDeconvDecoder(output_shape=input_shape,
                                                                latent_dim=encoder_latent_dim, 
                                                                nbr_attention_slot=nbr_attention_slot,
                                                                net_depth=decoder_nbr_layer,
                                                                conv_dim=decoder_conv_dim)

        body = BetaVAE(beta=beta,
                       encoder=encoder,
                       decoder=decoder,
                       input_shape=input_shape,
                       latent_dim=latent_dim,
                       nbr_attention_slot=nbr_attention_slot,
                       NormalOutputDistribution=NormalOutputDistribution,
                       constrainedEncoding=constrainedEncoding,
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