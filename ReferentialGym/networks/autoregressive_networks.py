import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision

import copy
import math
from numbers import Number 
from functools import partial

import numpy as np 
import cv2

from .networks import FCBody, ConvolutionalBody, MHDPA_RN, layer_init, ConvolutionalMHDPABody
from .residual_networks import ModelResNet18, ModelResNet18AvgPooled


class Distribution(object) :
    def sample(self) :
        raise NotImplementedError

    def log_prob(self,values) :
        raise NotImplementedError


class Bernoulli(Distribution) :
    def __init__(self, probs) :
        self.probs = probs

    def sample(self) :
        return torch.bernoulli(self.probs)

    def log_prob(self,values) :
        log_pmf = ( torch.stack( [1-self.probs, self.probs] ) ).log()

        return log_pmf.gather( 0, values.unsqueeze(0).long() ).squeeze(0)


class Normal(Distribution) :
    def __init__(self, mean, std) :
        self.mean = mean
        self.std = std

    def sample(self) :
        return torch.normal(self.mean, self.std)

    def log_prob(self,value) :
        var = self.std**2
        log_var = 2*math.log(self.std) if isinstance(self.std, Number) else 2*self.std.log() 

        return -( (value-self.mean) ** 2 ) / ( 2 * var) - 0.5*log_var - 0.5*math.log( 2*math.pi )


from .networks import addXYfeatures, conv, deconv, coordconv, coorddeconv, coord4deconv


class ResNetEncoder(ModelResNet18) :
    def __init__(self, input_shape, latent_dim=32, pretrained=False, nbr_layer=4, use_coordconv=False ) :
        super(ResNetEncoder,self).__init__(input_shape=input_shape, 
                                           feature_dim=2*latent_dim, 
                                           nbr_layer=nbr_layer, 
                                           pretrained=pretrained,
                                           use_coordconv=use_coordconv)
        
        self.input_shape = input_shape 
        self.latent_dim = latent_dim
        self.nbr_layer = nbr_layer

    def get_feature_shape(self):
        return self.latent_dim

    def encode(self, x) :
        out = super(ResNetEncoder,self).forward(x)
        return out

    def forward(self,x) :
        return self.encode(x)


class ResNetAvgPooledEncoder(ModelResNet18AvgPooled) :
    def __init__(self, input_shape, latent_dim=32, pretrained=False, nbr_layer=4, use_coordconv=False ) :
        super(ResNetAvgPooledEncoder,self).__init__(input_shape=input_shape, 
                                           feature_dim=2*latent_dim, 
                                           nbr_layer=nbr_layer, 
                                           pretrained=pretrained,
                                           use_coordconv=use_coordconv)
        
        self.input_shape = input_shape 
        self.latent_dim = latent_dim
        self.nbr_layer = nbr_layer

    def get_feature_shape(self):
        return self.latent_dim

    def encode(self, x) :
        out = super(ResNetAvgPooledEncoder,self).forward(x)
        return out

    def forward(self,x) :
        return self.encode(x)


class ResNetParallelAttentionEncoder(ModelResNet18) :
    def __init__(self, input_shape, latent_dim=10, nbr_attention_slot=10, pretrained=False, nbr_layer=4, use_coordconv=False) :
        super(ResNetParallelAttentionEncoder,self).__init__(input_shape=input_shape, 
                                           feature_dim=2*latent_dim, 
                                           nbr_layer=nbr_layer, 
                                           pretrained=pretrained,
                                           use_coordconv=use_coordconv)
        
        self.input_shape = input_shape 
        self.nbr_attention_slot = nbr_attention_slot
        self.latent_dim = latent_dim*nbr_attention_slot
        self.nbr_layer = nbr_layer

        self.spatialDim = self.feat_map_dim
        self.depthDim = self.feat_map_depth
        dimIn = self.depthDim+2
        dimOut = (self.spatialDim**2) * self.nbr_attention_slot

        self.feat_augmenter = addXYfeatures()
        self.recurrent_attention_computer = nn.GRU(input_size=dimIn,
                                                   hidden_size=dimOut,
                                                   num_layers=1,
                                                   bias=True,
                                                   batch_first=False,
                                                   dropout=0.0,
                                                   bidirectional=False)

        attention_slots_prior = torch.zeros((self.spatialDim, self.spatialDim, self.nbr_attention_slot))
        '''
        sqrt_as = math.floor(math.sqrt(self.nbr_attention_slot))
        spatial_step = self.spatialDim // sqrt_as
        for slot in range(self.nbr_attention_slot):
            attention_slots_prior[(slot%sqrt_as)*spatial_step:((slot%sqrt_as)+1)*spatial_step, \
             (slot//sqrt_as)*spatial_step:((slot//sqrt_as)+1)*spatial_step, \
             slot] = 1.0
        '''
        self.attention_slots_prior = nn.Parameter(attention_slots_prior.data)

    def get_feature_shape(self):
        return self.latent_dim

    def encode(self, x) :
        batch_size = x.size(0)
        self.features_map = super(ResNetParallelAttentionEncoder,self)._compute_feat_map(x)
        # batch x depth x dim x dim 
        augmented_features_map = self.feat_augmenter(features_map)
        recInput = augmented_features_map.transpose(1,3).contiguous().view((batch_size, -1, self.depthDim+2)).transpose(0,1)
        # dim*dim x batch x depth

        attention_weights = [self.attention_slots_prior.view(1,1,-1).repeat(1,batch_size,1)]
        for seq_idx in range(recInput.size(0)):
            attention_update, _ = self.recurrent_attention_computer(recInput[seq_idx].unsqueeze(0), attention_weights[-1])
            attention_weights.append(attention_weights[-1] + attention_update[-1].unsqueeze(0))

        attention_weights = [attention_weights[i].view((batch_size, self.spatialDim, self.spatialDim, self.nbr_attention_slot)) for i in range(len(attention_weights))]
        attention_weights = [ torch.sigmoid(attention) for attention in attention_weights]

        if False:
            ones = torch.ones(batch_size, self.spatialDim, self.spatialDim)
            residual = torch.zeros(batch_size, self.spatialDim, self.spatialDim)
            if features_map.is_cuda: 
                ones = ones.cuda()
                residual = residual.cuda()
            
            attention = []
            for slot in  range(self.nbr_attention_slot):
                attention.append( ((1-residual)*attention_weights[-1][...,slot]).unsqueeze(-1))
                residual = torch.min( ones, residual + attention[-1][...,-1]).detach()
            attention = torch.cat(attention, dim=-1)
        else:
            attention = attention_weights[-1]
        
        self.attention_weights = attention_weights
        
        hs = []
        for slot in range(self.nbr_attention_slot):
            attended_features_map = attention[...,slot].unsqueeze(1).repeat(1,features_map.size(1),1,1) * features_map 
            h = self._compute_features(attended_features_map)
            hs.append(h)
        hs = torch.cat(hs, dim=-1)  
        return hs

    def forward(self,x) :
        return self.encode(x)


class addXYSfeatures(nn.Module) :
    def __init__(self, nbr_attention_slot=10) :
        super(addXYSfeatures,self).__init__()
        self.nbr_attention_slot = nbr_attention_slot 
        self.fXYS = None

    def forward(self,x) :
        xsize = x.size()
        batch = xsize[0]
        if self.fXYS is None:
            # batch x depth x X x Y
            depth = xsize[1]
            sizeX = xsize[2]
            sizeY = xsize[3]
            stepX = 2.0/sizeX
            stepY = 2.0/sizeY

            fx = torch.zeros((1,1,sizeX,1))
            fy = torch.zeros((1,1,1,sizeY))
            
            vx = -1+0.5*stepX
            for i in range(sizeX):
                fx[:,:,i,:] = vx 
                vx += stepX
            vy = -1+0.5*stepY
            for i in range(sizeY):
                fy[:,:,:,i] = vy 
                vy += stepY
            fxy = fx.repeat(1,1,1,sizeY)
            fyx = fy.repeat(1,1,sizeX,1)
            fXY = torch.cat( [fxy,fyx], dim=1)
            # 1 x 2 x sizeX x sizeY

            fS = torch.zeros((1,self.nbr_attention_slot,sizeX,sizeY))
            self.fXYS = torch.cat([fXY,fS], dim=1)
            # 1 x 2+nbr_attention_slot x sizeX x sizeY            
            
        fXYS = self.fXYS.repeat(batch,1,1,1)
        if x.is_cuda : fXYS = fXYS.cuda()
        out = torch.cat( [x,fXYS], dim=1)
        return out 


class ResNetPHDPAEncoder(ModelResNet18) :
    def __init__(self, input_shape, latent_dim=10, nbr_attention_slot=10, pretrained=False, nbr_layer=4, use_coordconv=False) :
        super(ResNetPHDPAEncoder,self).__init__(input_shape=input_shape, 
                                           feature_dim=2*latent_dim, 
                                           nbr_layer=nbr_layer, 
                                           pretrained=pretrained,
                                           use_coordconv=use_coordconv)
        
        self.input_shape = input_shape 
        self.nbr_attention_slot = nbr_attention_slot
        self.latent_dim = latent_dim*nbr_attention_slot
        self.nbr_layer = nbr_layer

        self.spatialDim = self.feat_map_dim
        self.depthDim = self.feat_map_depth
        self.feat_augmenter = addXYSfeatures(nbr_attention_slot=self.nbr_attention_slot)
        self.attention_ponderer_depth_dim = self.depthDim+2+self.nbr_attention_slot
        self.attention_ponderer_nbr_entity = self.spatialDim*self.spatialDim
        self.attention_ponderer = MHDPA_RN(output_dim=None,
                                           depth_dim=self.attention_ponderer_depth_dim,
                                           nbrHead=1,#self.nbr_attention_slot, 
                                           nbrRecurrentSharedLayers=1, 
                                           nbrEntity=self.attention_ponderer_nbr_entity,  
                                           units_per_MLP_layer=256,
                                           interactions_dim=32,#256,
                                           dropout_prob=0.0)

    def get_feature_shape(self):
        return self.latent_dim

    def encode(self, x) :
        batch_size = x.size(0)
        self.features_map = super(ResNetPHDPAEncoder,self)._compute_feat_map(x)
        # batch x depth x dim x dim 
        augmented_features_map = self.feat_augmenter(features_map).view((batch_size, -1, self.spatialDim**2))
        # batch x depth+x+y+nbr_attention_slot (depht+2+nbr_attention_slot) x dim*dim 
        pondered_features_map = self.attention_ponderer(augx=augmented_features_map).view((batch_size, -1, self.spatialDim, self.spatialDim))
        # batch x depth+x+y+nbr_attention_slot (depht+2+nbr_attention_slot) x dim x dim 
        attention_weights = torch.cat( [pondered_features_map[:, -slot, ...].unsqueeze(1).unsqueeze(-1) for slot in range(self.nbr_attention_slot)], dim=-1)
        attention_weights = F.softmax(attention_weights, dim=-1)

        self.attention_weights = [attention_weights]

        hs = []
        for slot in range(self.nbr_attention_slot):
            attended_features_map = attention_weights[...,slot].repeat(1,self.depthDim,1,1)* features_map 
            h = self._compute_features(attended_features_map)
            hs.append(h)
        hs = torch.cat(hs, dim=-1)  
        return hs

    def forward(self,x) :
        return self.encode(x)


class Decoder(nn.Module) :
    def __init__(self, output_shape=[3, 64, 64], net_depth=3, latent_dim=32, conv_dim=64) :
        super(Decoder,self).__init__()

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.output_shape = output_shape
        self.net_depth = net_depth
        self.latent_dim = latent_dim 
        self.conv_dim = conv_dim

        self.dcs = []
        outd = self.conv_dim*(2**self.net_depth)
        ind= self.latent_dim
        k = 4
        dim = k
        pad = 1
        stride = 2
        self.fc = coorddeconv( ind, outd, k, stride=1, padding=0, batchNorm=False)
        
        for i in reversed(range(self.net_depth)) :
            ind = outd
            outd = self.conv_dim*(2**i)
            self.dcs.append( coorddeconv( ind, outd,k,stride=stride,padding=pad) )
            self.dcs.append( nn.ReLU() )
            dim = k-2*pad + stride*(dim-1)
            print('Decoder: layer {} : dim {}.'.format(i, dim))
        self.dcs = nn.Sequential( *self.dcs) 
            
        ind = outd
        self.img_depth= self.output_shape[0]
        outd = self.img_depth
        outdim = self.output_shape[1]
        indim = dim
        pad = 0
        stride = 1
        k = outdim +2*pad -stride*(indim-1)
        self.dcout = coorddeconv( ind, outd, k, stride=stride, padding=pad, batchNorm=False)
        
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = F.leaky_relu( self.fc(z), 0.05)
        out = F.leaky_relu( self.dcs(out), 0.05)
        out = torch.sigmoid( self.dcout(out))
        return out

    def forward(self,z) :
        return self.decode(z)

class DeconvolutionalBody(nn.Module) :
    def __init__(self,
                 input_shape, 
                 output_shape=[3, 64, 64], 
                 channels=[3,3], 
                 kernel_sizes=[1], 
                 strides=[1], 
                 paddings=[0],
                 dropout=0.0, 
                 non_linearities=[nn.LeakyReLU],
                 use_coordconv=None) :
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: Integer, dimensions of the input.
        :param output_shape: List[Integer] desired dimensions of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param fc_hidden_units: list of number of neurons per fully-connected 
                hidden layer following the convolutional layers.
        :param dropout: dropout probability to use.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        :param use_coordconv: None or Int specifying the type of coord convolutional layers to use, either 2 or 4.
        '''
        super(DeconvolutionalBody,self).__init__()
        original_deconv_fn = nn.ConvTranspose2d
        if use_coordconv is not None:
            if use_coordconv == 2: 
                original_deconv_fn = coorddeconv
            elif use_coordconv == 4:
                original_deconv_fn = coord4deconv
            else:
                raise NotImplementedError

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.dropout = dropout
        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities] * (len(channels) - 1)
        else:
            while len(self.non_linearities) <= (len(channels) - 1):
                self.non_linearities.append(self.non_linearities[0])

        self.input_shape = input_shape[0]
        self.output_shape = output_shape
        
        self.dcs = []
        outd = self.input_shape//2
        ind= self.input_shape
        k = 2 #4
        dim = k
        pad = 1
        stride = 2
        self.fc = coorddeconv( ind, outd, k, stride=1, padding=0, batchNorm=False)
        
        """
        for i in reversed(range(self.net_depth)) :
            ind = outd
            outd = self.conv_dim*(2**i)
            self.dcs.append( coorddeconv( ind, outd,k,stride=stride,padding=pad) )
            self.dcs.append( nn.ReLU() )
            dim = k-2*pad + stride*(dim-1)
            print('Decoder: layer {} : dim {}.'.format(i, dim))
        self.dcs = nn.Sequential( *self.dcs) 
        """ 

        self.dcs = []
        in_ch = outd
        for idx, (cfg, k, s, p) in enumerate(zip(channels[1:], kernel_sizes, strides, paddings)):
            deconv_fn = original_deconv_fn
            if isinstance(cfg, str) and cfg == 'MP':
                if isinstance(k, str):
                    assert(k=="Full")
                    k = dim
                    channels[idx+1] = in_ch
                layer = nn.MaxPool2d(kernel_size=k, stride=s)
                self.dcs.append(layer)
                # Update of the shape of the input-image, following Conv:
                dim = (dim-k)//s+1
                print(f"Dim: {dim}")
            else:
                add_non_lin = True
                add_dp = (self.dropout > 0.0)
                dropout = self.dropout
                add_bn = False
                add_ln = False
                if isinstance(cfg, str) and 'NoNonLin' in cfg:
                    add_non_lin = False
                    cfg = cfg.replace('NoNonLin', '') 
                if isinstance(cfg, str) and 'Coord2' in cfg:
                    deconv_fn = coordconv
                    cfg = cfg.replace('Coord2', '') 
                elif isinstance(cfg, str) and 'Coord4' in cfg:
                    deconv_fn = coord4conv
                    cfg = cfg.replace('Coord4', '') 
                
                if isinstance(cfg, str) and '_DP' in cfg:
                    add_dp = True
                    cfg = cfg.split('_DP')
                    dropout = float(cfg[-1])
                    cfg = cfg[0] 
                    # Assumes 'YX_DPZ'
                    # where Y may be BN/LN/nothing
                    # and X is an integer
                    # and Z is the float dropout value.
                
                if isinstance(cfg, str) and 'BN' in cfg:
                    add_bn = True
                    cfg = int(cfg[2:])
                    channels[idx+1] = cfg
                    # Assumes 'BNX' where X is an integer...
                elif isinstance(cfg, str) and 'LN' in cfg:
                    add_ln = True
                    cfg = int(cfg[2:])
                    channels[idx+1] = cfg
                    # Assumes 'LNX' where X is an integer...
                elif isinstance(cfg, str):
                    cfg = int(cfg)
                    channels[idx+1] = cfg
                    
                layer = deconv_fn(in_ch, cfg, kernel_size=k, stride=s, padding=p, bias=not(add_bn)) 
                layer = layer_init(layer, w_scale=math.sqrt(2))
                in_ch = cfg
                self.dcs.append(layer)
                if add_bn:
                    self.dcs.append(nn.BatchNorm2d(in_ch))
                if add_ln:
                    # Layer Normalization:
                    # solely about the last dimension of the 4D tensor, i.e. channels...
                    # TODO: It might be necessary to have the possibility to apply this 
                    # normalization over the other dimensions, i.e. width x height...
                    self.dcs.append(nn.LayerNorm(in_ch))
                if add_dp:
                    self.dcs.append(nn.Dropout2d(p=dropout))
                if add_non_lin:
                    #self.cnn.append(self.non_linearities[idx](inplace=True))
                    self.dcs.append(self.non_linearities[idx]())
                # Update of the shape of the input-image, following Conv:
                #dim = (dim-k+2*p)//s+1
                dim = k-2*pad + stride*(dim-1)
                print(f"Dim: {dim}")
        self.dcs = nn.Sequential(*self.dcs)

        ind = in_ch
        self.img_depth = self.output_shape[0]
        outdim = self.output_shape[1]
        indim = dim
        pad = 0
        stride = 1
        k = outdim +2*pad -stride*(indim-1)
        self.dcout = coorddeconv( ind, self.img_depth, k, stride=stride, padding=pad, batchNorm=False)
    
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = F.leaky_relu( self.fc(z), 0.05)
        out = F.leaky_relu( self.dcs(out), 0.05)
        out = self.dcout(out)
        return out

    def forward(self,z) :
        return self.decode(z)

    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.output_shape


class BroadcastingDecoder(nn.Module) :
    def __init__(self, output_shape=[3, 64, 64], 
                       net_depth=3, 
                       kernel_size=3, 
                       stride=1, 
                       padding=1, 
                       latent_dim=32, 
                       conv_dim=64):
        super(BroadcastingDecoder,self).__init__()

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.output_shape = output_shape
        self.net_depth = net_depth
        self.latent_dim = latent_dim 
        self.conv_dim = conv_dim

        self.dcs = []
        dim = self.output_shape[-1]
        outd = self.conv_dim
        ind= self.latent_dim
        
        for i in range(self.net_depth) :
            
            if i == self.net_depth-1: 
                outd = self.output_shape[0]

            if i == 0: 
                layer = layer_init(coordconv( ind, outd, kernel_size, stride=stride, padding=padding), w_scale=1e-3)
            else:
                layer = layer_init(nn.Conv2d(ind, outd, kernel_size=kernel_size, stride=stride, padding=padding), w_scale=1e-3)
            
            self.dcs.append(layer)

            if i != self.net_depth-1: 
                self.dcs.append( nn.ReLU() )
                #self.dcs.append( nn.LeakyReLU(0.05) )
            
            ind = outd 
            dim = (dim-kernel_size+2*padding)//stride+1
            print('BroadcastingDecoder: layer {} : dim {}.'.format(i, dim))
        
        self.dcs = nn.Sequential( *self.dcs) 
                
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = z.repeat(1, 1, self.output_shape[-2], self.output_shape[-1])
        out = self.dcs(out)
        #out = torch.sigmoid(out)
        return out

    def forward(self,z) :
        return self.decode(z)


class BroadcastingDeconvDecoder(nn.Module) :
    def __init__(self, output_shape=[3, 64, 64], net_depth=3, latent_dim=32, conv_dim=64) :
        super(BroadcastingDeconvDecoder,self).__init__()

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.output_shape = output_shape
        self.net_depth = net_depth
        self.latent_dim = latent_dim 
        self.conv_dim = conv_dim

        self.dcs = []
        outd = self.conv_dim*(2**self.net_depth)
        ind= self.latent_dim
        k = 4
        dim = k
        pad = 1
        stride = 2

        outd = ind 
        for i in reversed(range(self.net_depth)) :
            ind = outd
            outd = self.conv_dim*(2**i)
            self.dcs.append( coorddeconv( ind, outd,k,stride=stride,padding=pad) )
            self.dcs.append( nn.ReLU() )
            dim = k-2*pad + stride*(dim-1)
            print('BroadcastingDeconvDecoder: layer {} : dim {}.'.format(i, dim))
        self.dcs = nn.Sequential( *self.dcs) 
            
        ind = outd
        self.img_depth= self.output_shape[0]
        outd = self.img_depth
        outdim = self.output_shape[1]
        indim = dim
        pad = 0
        stride = 1
        k = outdim +2*pad -stride*(indim-1)
        self.dcout = coorddeconv( ind, outd, k, stride=stride, padding=pad, batchNorm=False)
        
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = z.repeat(1, 1, 4, 4)
        out = self.dcs(out)
        out = torch.sigmoid( self.dcout(out))
        return out

    def forward(self,z) :
        return self.decode(z)


class ParallelAttentionBroadcastingDeconvDecoder(nn.Module) :
    def __init__(self, output_shape=[3, 64, 64], net_depth=3, latent_dim=32, nbr_attention_slot=10, conv_dim=64) :
        super(ParallelAttentionBroadcastingDeconvDecoder,self).__init__()

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.output_shape = output_shape
        self.net_depth = net_depth
        self.latent_dim = latent_dim
        self.nbr_attention_slot = nbr_attention_slot
        self.conv_dim = conv_dim

        self.dcs = []
        outd = self.conv_dim*(2**self.net_depth)
        ind= self.latent_dim
        k = 4
        dim = k
        pad = 1
        stride = 2

        outd = ind 
        for i in reversed(range(self.net_depth)) :
            ind = outd
            outd = self.conv_dim*(2**i)
            self.dcs.append( coorddeconv( ind, outd,k,stride=stride,padding=pad) )
            self.dcs.append( nn.ReLU() )
            dim = k-2*pad + stride*(dim-1)
            print('ParallelAttentionBroadcastingDeconvDecoder: layer {} : dim {}.'.format(i, dim))
        self.dcs = nn.Sequential( *self.dcs) 
            
        ind = outd*self.nbr_attention_slot
        self.img_depth = self.output_shape[0]
        outd = self.img_depth
        outdim = self.output_shape[1]
        indim = dim
        pad = 0
        stride = 1
        k = outdim +2*pad -stride*(indim-1)
        self.dcout = coorddeconv( ind, outd, k, stride=stride, padding=pad, batchNorm=False)
        
    def decode(self, z) :
        zs = torch.chunk(z, self.nbr_attention_slot, dim=1)
        parallel_outputs = []
        for slot in range(self.nbr_attention_slot):
            z = zs[slot]
            z = z.view( z.size(0), z.size(1), 1, 1)
            out = z.repeat(1, 1, 4, 4)
            out = F.leaky_relu( self.dcs(out), 0.05)
            parallel_outputs.append(out)
        parallel_outputs = torch.cat(parallel_outputs, dim=1)
        output = torch.sigmoid( self.dcout(parallel_outputs))
        return output

    def forward(self,z) :
        return self.decode(z)

# Very Important:
# Making the TotalCorrelationDiscriminator an object
# rather than an nn.Module prevents the optimization_module
# from gathering its parameters and optimizing for them.
# Rather, optimization is handled internally whenever the 
# mode is training.
class TotalCorrelationDiscriminator(object):
    def __init__(self, VAE):
        self.latent_dim = VAE.latent_dim
        #tc_discriminator_hidden_units = [2*self.latent_dim]*4+[2]
        # Sizes from 2*1000 to 6*1000 were found valuable in Appendix A:
        tc_discriminator_hidden_units = [1000]*4+[2]
        self.discriminator = FCBody(
            state_dim=self.latent_dim,
            hidden_units=tc_discriminator_hidden_units,
            non_linearities=[nn.LeakyReLU]
        )
        
        # Hyperparameters of the optimiser are found in 
        # [Kim and Mnih, 2018, "Disentangling by Factorising"]'s
        # appendix A.
        # Expecting batch size of 64.
        # ... remembering that the independence
        # testing trick is as good as the batch size
        # is sufficiently large (compared to the size of the dataset).  
        self.optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=1e-4,
            betas=[0.5,0.9],
        )
        self.need_weights_update = False

        print("TotalCorrelationDiscriminator:")
        print(self.discriminator)

    def __call__(self, z):
        if self.need_weights_update:
            self.need_weights_update = False
            self.optimizer.zero_grad()
            self.TC_loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if z.is_cuda: 
            self.discriminator = self.discriminator.cuda()
        return self.discriminator(z)

    def update(self, z, train=True):
        z = z.detach()
        z1, z2 = z.chunk(2, dim=0)
        
        zeros = torch.zeros(z1.size(0)).long().to(z1.device)
        ones = torch.ones(z2.size(0)).long().to(z2.device)
        
        pz2 = self.permutate_latents(z2)
        Dz1 = self.discriminator(z1)
        Dpz2 = self.discriminator(pz2)
        # TC losses:
        tc_l11 = 0.5*F.cross_entropy(input=Dz1, target=zeros, reduction='none')
        # (b1, )
        tc_l12 = 0.5*F.cross_entropy(input=Dpz2, target=ones, reduction='none')
        # (b2, )
        
        zeros = torch.zeros(z2.size(0)).long().to(z2.device)
        ones = torch.ones(z1.size(0)).long().to(z1.device)
        
        pz1 = self.permutate_latents(z1)
        Dz2 = self.discriminator(z2)
        Dpz1 = self.discriminator(pz1)

        # TC losses:
        tc_l21 = 0.5*F.cross_entropy(input=Dz2, target=zeros, reduction='none')
        # (b1, )
        tc_l22 = 0.5*F.cross_entropy(input=Dpz1, target=ones, reduction='none')
        # (b2, )
        
        # TC loss:
        tc_loss1 = tc_l11 + tc_l22
        tc_loss2 = tc_l12 + tc_l21
        
        self.TC_loss = torch.cat([tc_loss1, tc_loss2])
        # (b, )

        if train: self.step()

        # Compute discriminator accuracy:
        Dz = torch.cat([Dz1, Dz2], dim=0)
        Dpz = torch.cat([Dpz1, Dpz2], dim=0)
        probDz = F.softmax(Dz.detach(), dim=1)[...,:1]
        probDpz = F.softmax(Dpz.detach(), dim=1)[...,1:]
        discr_acc = (torch.cat([probDz,probDpz],dim=0) >= 0.5).sum().float().div(2*probDz.size(0))
        self.discr_acc = discr_acc.cpu().unsqueeze(0)

        return self.TC_loss, self.discr_acc

    def step(self):
        #self.TC_loss.mean().backward()
        self.need_weights_update = True
       
    def permutate_latents(self, z):
        assert(z.dim() == 2)
        batch_size, latent_dim = z.size()
        pz = list()
        for lz in torch.split(z, split_size_or_sections=1, dim=1):
            b_perm = torch.randperm(batch_size).to(z.device)
            p_lz = lz[b_perm]
            pz.append(p_lz)
        pz = torch.cat(pz, dim=1)
        return pz


class BetaVAE(nn.Module) :
    def __init__(self, beta=1e4, 
                       encoder=None,
                       decoder=None,
                       latent_dim=32,
                       nbr_attention_slot=None,
                       input_shape=[3, 64, 64], 
                       NormalOutputDistribution=True,
                       maxEncodingCapacity=1000,
                       nbrEpochTillMaxEncodingCapacity=4,
                       constrainedEncoding=True,
                       observation_sigma=0.05,
                       factor_vae_gamma=0.0):
        super(BetaVAE,self).__init__()

        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self.observation_sigma = observation_sigma
        self.latent_dim = latent_dim
        self.nbr_attention_slot = nbr_attention_slot
        self.input_shape = input_shape
        self.NormalOutputDistribution = NormalOutputDistribution
        self.factor_vae_gamma = factor_vae_gamma

        if self.factor_vae_gamma > 0.0:
            '''
            self.tc_discriminator_hidden_units = tuple([2*self.latent_dim]*4+[2])
            self.tc_discriminator = FCBody(state_dim=self.latent_dim,
                                           hidden_units=tc_discriminator_hidden_units,
                                           gate=F.leaky_relu)
            '''
            self.tc_discriminator = TotalCorrelationDiscriminator(self)
        else:
            self.tc_discriminator = None 

        self.EncodingCapacity = 0.0
        self.EncodingCapacityStep = None
        self.maxEncodingCapacity = maxEncodingCapacity
        self.constrainedEncoding = constrainedEncoding
        self.nbrEpochTillMaxEncodingCapacity = nbrEpochTillMaxEncodingCapacity
        
        self.increaseEncodingCapacity = True
        if self.constrainedEncoding:
            nbritperepoch = 300
            print('ITER PER EPOCH : {}'.format(nbritperepoch))
            nbrepochtillmax = self.nbrEpochTillMaxEncodingCapacity
            nbrittillmax = nbrepochtillmax * nbritperepoch
            print('ITER TILL MAX ENCODING CAPACITY : {}'.format(nbrittillmax))
            self.EncodingCapacityStep = self.maxEncodingCapacity / nbrittillmax        

    def get_feature_shape(self):
        return self.latent_dim
    
    def _compute_feature_shape(self, input_dim=None, nbr_layer=None):
        return self.encoder._compute_feature_shape(input_dim=input_dim, nbr_layer=nbr_layer)
        
    def reparameterize(self, mu,log_var) :
        eps = torch.randn( (mu.size()[0], mu.size()[1]) )
        if mu.is_cuda:  eps = eps.cuda()
        z = mu + eps * torch.exp( log_var ).sqrt()
        return z

    def forward(self,x) :
        self.x = x 
        self.h = self.encoder(self.x)
        self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
        self.z = self.reparameterize(self.mu, self.log_var)
        self.out = self.decoder(self.z)
        return self.out, self.mu, self.log_var
    
    def encode(self,x) :
        self.h = self.encoder(x)
        self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
        return self.mu

    def encodeZ(self,x) :
        self.x = x 
        self.h = self.encoder(self.x)
        self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
        self.z = self.reparameterize(self.mu, self.log_var)        
        return self.z, self.mu, self.log_var

    def decode(self, z):
        return self.decoder(z)

    def _forward(self,x=None,evaluation=False,fixed_latent=None,data=None) :
        if data is None and x is not None :
            if evaluation :
                self.z, self.mu, self.log_var = self.encodeZ(x) 
            else :
                self.x = x 
                self.h = self.encoder(self.x)
                self.mu, self.log_var = torch.chunk(self.h, 2, dim=1 )
                self.z = self.reparameterize(self.mu, self.log_var)
                self.VAE_output = self.decoder(self.z)
        elif data is not None :
            self.mu, self.log_var = data 
            self.z = self.reparameterize(self.mu, self.log_var)
            if not(evaluation) :
                self.VAE_output = self.decoder(self.z)

        self.batch_size = self.z.size()[0]
        if fixed_latent is not None :
            idx = fixed_latent[0]
            val = fixed_latent[1]
            self.mu = self.mu.cpu().data 
            self.mu[:,idx] = val
            if next(self.parameters()).is_cuda : self.mu = self.mu.cuda()
            self.z = self.reparameterize(self.mu,self.log_var)
            
        return self.mu, self.log_var, self.VAE_output  

    def get_feat_map(self):
        return self.encoder.get_feat_map()

    def compute_loss(self,x=None,
                        fixed_latent=None,
                        data=None,
                        evaluation=False,
                        observation_sigma=None) :
        if x is None: 
            if self.x is not None:
                x = self.x 
            else:
                raise NotImplementedError

        gtx = x 
        xsize = x.size()
        self.batch_size = xsize[0]
        
        self._forward(x=x,fixed_latent=fixed_latent,data=data,evaluation=evaluation)
        
        if evaluation :
            self.VAE_output = gtx 

        output_dict = dict()
        #--------------------------------------------------------------------------------------------------------------
        # VAE loss :
        #--------------------------------------------------------------------------------------------------------------
        # Reconstruction loss :
        self.binary_crossentropy = F.binary_cross_entropy_with_logits(self.VAE_output, gtx, reduction='sum').div(self.batch_size)
        
        self.VAE_output = self.VAE_output.sigmoid()

        if observation_sigma is not None:
            self.observation_sigma = observation_sigma
        if self.NormalOutputDistribution:
            #Normal :
            self.neg_log_lik = -Normal(self.VAE_output, self.observation_sigma).log_prob( gtx)

            # http://ruishu.io/2018/03/14/vae/ :
            # Since the observation model is assumed as gaussian, we should sample from it, with the dedicated fixed std:
            '''
            VAE_output_distr = torch.distributions.Normal(loc=self.VAE_output, scale=self.observation_sigma)
            rsample_VAE_output = VAE_output_distr.rsample()
            self.VAE_loss =  torch.sum( F.mse_loss(input=rsample_VAE_output, target=gtx, reduction='none').view( self.batch_size, -1), dim=1)
            '''
            # Makes for a sharper image, since there is no noise...
            self.VAE_loss =  torch.sum( F.mse_loss(input=self.VAE_output, target=gtx, reduction='none').view( self.batch_size, -1), dim=1)
            output_dict.update({'MSE_reconst_loss': self.VAE_loss})
        else:
            #Bernoulli :
            #self.neg_log_lik = -torch.distributions.Bernoulli( torch.sigmoid(self.VAE_output) ).log_prob( gtx )
            #self.neg_log_lik = -torch.distributions.Bernoulli(self.VAE_output).log_prob( gtx )
            self.neg_log_lik = -Bernoulli(self.VAE_output).log_prob( gtx )
            #self.VAE_loss = self.reconst_loss
            self.VAE_loss = self.binary_crossentropy

        self.reconst_loss = torch.sum( self.neg_log_lik.view( self.batch_size, -1), dim=1)        
        
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        # KL Divergence :
        self.true_kl_divergence = 0.5 * (self.mu**2 + torch.exp(self.log_var) - self.log_var -1)
        
        if self.EncodingCapacityStep is None :
            self.kl_divergence = torch.sum(self.true_kl_divergence, dim=1)
            self.kl_divergence_regularized =  torch.abs( torch.sum(self.true_kl_divergence, dim=1) - self.EncodingCapacity ) 
            self.VAE_loss = self.VAE_loss + self.beta*self.kl_divergence
        else:
            self.kl_divergence_regularized =  torch.abs( torch.sum(self.true_kl_divergence, dim=1) - self.EncodingCapacity ) 
            self.kl_divergence =  torch.sum( self.true_kl_divergence, dim=1 )
            self.VAE_loss = self.VAE_loss + self.beta * self.kl_divergence_regularized
            
            if self.increaseEncodingCapacity and self.training:
                self.EncodingCapacity += self.EncodingCapacityStep
            if self.EncodingCapacity >= self.maxEncodingCapacity :
                self.increaseEncodingCapacity = False 
        #--------------------------------------------------------------------------------------------------------------
        output_dict.update({'VAE_loss': self.VAE_loss, 
                            'reconst_loss': self.reconst_loss,
                            'binary_crossentropy': self.binary_crossentropy.unsqueeze(0),
                            'neg_log_lik': self.neg_log_lik, 
                            'kl_div_reg': self.kl_divergence_regularized, 
                            'kl_div': self.true_kl_divergence,
        })

        #for muidx in range(self.mu.shape[-1]):
        #    output_dict[f'mu_{muidx}'] = self.mu[:,muidx:muidx+1]
        #    output_dict[f'log_var_{muidx}'] = self.log_var[:,muidx:muidx+1]
        output_dict[f'mu'] = self.mu.mean(dim=-1, keepdim=True)
        output_dict[f'logvar'] = self.log_var.mean(dim=-1, keepdim=True)
        
        #--------------------------------------------------------------------------------------------------------------
        # FactorVAE losses:
        if self.factor_vae_gamma > 0.0:
            # VAE TC loss:
            Dz = self.tc_discriminator(self.z)
            self.VAE_TC_loss = (Dz[:,0] - Dz[:,1])
            #(b, )            

            soft_Dz = torch.softmax(Dz, dim=-1)
            # logit 0: probability for the input to come from the model.
            # logit 1: probability for the input to come from the permutated generator.
            # ==> When the model has a highly modular output, the discriminator should
            # not be able to discriminate with the permutated generator, and therefore output
            # high probability for logit 1 (and low for logit 0).
            # i.e. the model has modularity as high as logit 1 is high and logit 0 is low :
            self.vae_modularity = (1+soft_Dz[:,1]-soft_Dz[:,0])/2
            #PREVIOUSLY:
            #self.vae_modularity = 1-(soft_Dz[:,0]-soft_Dz[:,1])

            # TC Discriminator:
            z = self.z.detach()
            if z.size(0) > 1:
                self.TC_loss, self.discr_acc = self.tc_discriminator.update(z, train=self.training) 
            else:
                self.TC_loss = torch.zeros(1).to(z.device)
                self.discr_acc = 0.5*torch.ones(1).to(z.device)

            output_dict.update({'TC_loss': self.TC_loss,
                                'VAE_TC_loss': self.VAE_TC_loss,
                                'Modularity': self.vae_modularity,
                                'DiscriminatorAccuracy': self.discr_acc,
                                'VAE_loss': self.VAE_loss+self.factor_vae_gamma*self.VAE_TC_loss}
                                )

        return output_dict 


class UNetBlock(nn.Module):
    def __init__(self, 
                 in_channel, 
                 out_channel, 
                 upsample=True, 
                 interpolate=False,
                 interpolation_factor=2,
                 batch_norm=False):
        super(UNetBlock, self).__init__()

        self.upsample = upsample
        self.interpolate = interpolate
        self.interpolation_factor = interpolation_factor

        self.norm = partial( nn.InstanceNorm2d, affine=True, track_running_stats=False)
        if batch_norm:
            self.norm = nn.BatchNorm2d

        if self.upsample:
            self.layers = nn.Sequential(
                layer_init(nn.Conv2d(in_channel*2,in_channel, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
                self.norm(num_features=in_channel),
                nn.ReLU(),
                layer_init(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
                self.norm(num_features=out_channel),
                nn.ReLU()
            )
        else:
            self.layers = nn.Sequential(
                layer_init(nn.Conv2d(in_channel,out_channel, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
                self.norm(num_features=out_channel),
                nn.ReLU(),
                layer_init(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
                self.norm(num_features=out_channel),
                nn.ReLU()
            )


    def forward(self, x):
        out = self.layers(x)
        interout = out 
        if self.interpolate:
            interout = F.interpolate(out, scale_factor=self.interpolation_factor)
        return interout, out


class UNet(nn.Module):
    def __init__(self, 
                 input_shape,
                 in_channel,
                 out_channel, 
                 basis_nbr_channel=32, 
                 block_depth=3, 
                 batch_norm=False):
        super(UNet, self).__init__()

        self.input_shape = input_shape
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.basis_nbr_channel = basis_nbr_channel
        self.block_depth = block_depth

        self.downsampling_blocks = nn.ModuleList()
        running_nbr_channel = self.basis_nbr_channel
        
        interpolation = True
        interpolation_factor=0.5
        spatialDim = self.input_shape[-1]
        for ib in range(self.block_depth):
            b = UNetBlock(in_channel, 
                          running_nbr_channel,
                          upsample=False,
                          interpolate=interpolation,
                          interpolation_factor=interpolation_factor,
                          batch_norm=batch_norm)
            self.downsampling_blocks.append(b)
            in_channel = running_nbr_channel
            running_nbr_channel *= 2
            spatialDim /= 2

        running_nbr_channel //= 2
        
        '''
        self.mid_block = nn.Sequential(
            layer_init(nn.Conv2d(running_nbr_channel,running_nbr_channel*2, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
            nn.InstanceNorm2d(num_features=running_nbr_channel*2, affine=True, track_running_stats=False),
            nn.ReLU(),
            layer_init(nn.Conv2d(running_nbr_channel*2, running_nbr_channel, kernel_size=3, stride=1, padding=1, bias=False), w_scale=1e-3),
            nn.InstanceNorm2d(num_features=running_nbr_channel, affine=True, track_running_stats=False),
            nn.ReLU()
        )
        '''
        self.mid_block = nn.Sequential(
            layer_init(nn.Linear(int(running_nbr_channel*(spatialDim**2)), 128), w_scale=1e-3),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128), w_scale=1e-3),
            nn.ReLU(),
            layer_init(nn.Linear(128, int(running_nbr_channel*(spatialDim**2))), w_scale=1e-3),
            nn.ReLU(),
        )
        
        in_channel = running_nbr_channel*2
        running_nbr_channel //=2

        self.upsampling_blocks = nn.ModuleList()
        interpolation_factor = 2
        for ib in range(self.block_depth):
            b = UNetBlock(in_channel, 
                          running_nbr_channel,
                          upsample=False,
                          interpolate=interpolation,
                          interpolation_factor=interpolation_factor,
                          batch_norm=batch_norm)
            self.upsampling_blocks.append(b)
            in_channel = 2 * running_nbr_channel
            running_nbr_channel //= 2
        running_nbr_channel *= 2

        self.final_conv = layer_init(nn.Conv2d(running_nbr_channel, self.out_channel, kernel_size=1, stride=1, padding=0, bias=True), w_scale=1e-3)

    def forward(self, x):
        skipxs = list()
        for bidx, block in enumerate(self.downsampling_blocks):
            x, xout = block(x)
            skipxs.append(xout)

        # flattening non-skip output:
        xsize = x.size()
        flat_x = x.flatten(start_dim=1)
        x = self.mid_block(flat_x)
        x = x.view(xsize)
        '''
        x = self.mid_block(x)
        '''
        
        x = F.interpolate(x, scale_factor=2)
        
        for block, skipx in zip(self.upsampling_blocks, reversed(skipxs)):
            xin = torch.cat([skipx, x], dim=1)
            x, xout = block(xin)

        # consuming the non-interpolated output:
        y = self.final_conv(xout)

        return y 



class AttentionNetwork(nn.Module):
    def __init__(self, 
                 input_shape,
                 in_channel, 
                 attention_basis_nbr_channel=32,
                 attention_block_depth=3):
        super(AttentionNetwork, self).__init__()
        self.input_shape = input_shape
        self.in_channel = input_shape[0]+1
        '''
        self.unet = UNet(input_shape=self.input_shape,
                         in_channel=self.in_channel,
                         out_channel=1,
                         basis_nbr_channel=attention_basis_nbr_channel,
                         block_depth=attention_block_depth)
        '''
        nChannel = attention_basis_nbr_channel
        self.unet = [\
            nn.Conv2d(self.in_channel, nChannel, kernel_size=3, stride=1, padding=1), \
            nn.ReLU(), \
            nn.BatchNorm2d(nChannel)]

        for i in range(attention_block_depth):
            self.unet.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.unet.append(nn.ReLU())
            self.unet.append( nn.BatchNorm2d(nChannel))
        
        self.unet.append(nn.Conv2d(nChannel, 1, kernel_size=1, stride=1, padding=0))
        self.unet.append(nn.BatchNorm2d(1))

        self.unet = nn.Sequential(*self.unet)

    def forward(self, x, logscope):
        xin = torch.cat([x, logscope], dim=1)
        xout = self.unet(xin)
        # log( m_k ) = log( s_{k-1} * sigmoid(unet(xin)) )
        logmask = logscope+F.logsigmoid(xout)
        # log( s_k ) = log( s_{k-1} * (1-sigmoid(unet(xin))) ) = log(s_{k-1}) + log( 1-sigmoid(unet(xin))) (==sigmoid(-unet(xin)))
        nlogscope = logscope+F.logsigmoid(-xout)
        return logmask, nlogscope

class ParallelAttentionNetwork(nn.Module):
    def __init__(self, 
                 input_shape,
                 in_channel, 
                 nbr_attention_slot=10,
                 attention_basis_nbr_channel=32,
                 attention_block_depth=3):
        super(ParallelAttentionNetwork, self).__init__()
        self.input_shape = input_shape
        self.in_channel = input_shape[0]+1
        self.nbr_attention_slot = nbr_attention_slot
        nChannel = attention_basis_nbr_channel
        self.net = [\
            nn.Conv2d(self.in_channel, nChannel, kernel_size=3, stride=1, padding=1), \
            nn.ReLU(), \
            nn.BatchNorm2d(nChannel)]

        for i in range(attention_block_depth):
            self.net.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.net.append(nn.ReLU())
            self.net.append( nn.BatchNorm2d(nChannel))
        
        self.net.append(nn.Conv2d(nChannel, self.nbr_attention_slot-1, kernel_size=1, stride=1, padding=0))
        self.net.append(nn.BatchNorm2d(self.nbr_attention_slot-1))

        self.net = nn.Sequential(*self.net)

    def forward(self, x, logscope):
        batch_size = x.size(0)
        xin = torch.cat([x, logscope], dim=1)
        xout = self.net(xin)

        logmasks = torch.empty(batch_size, self.nbr_attention_slot, *self.input_shape[-2:]).to(x.device)
        logscopes = torch.empty(batch_size, self.nbr_attention_slot, *self.input_shape[-2:]).to(x.device)
        logscopes[:,0:] = logscope 

        for slot in range(self.nbr_attention_slot):
            if slot < self.nbr_attention_slot-1:
                x = xout[:,slot:slot+1,...]
                ls = logscopes[:,slot:slot+1,...]  
                # log( m_k ) = log( s_{k-1} * sigmoid(unet(xin)) )
                #logmask = logscope+F.logsigmoid(xout)
                logmask = ls+F.logsigmoid(x)
                # log( s_k ) = log( s_{k-1} * (1-sigmoid(unet(xin))) ) = log(s_{k-1}) + log( 1-sigmoid(unet(xin))) (==sigmoid(-unet(xin)))
                #nlogscope = logscope+F.logsigmoid(-xout)
                nls = ls+F.logsigmoid(-x)

                logmasks[:,slot:slot+1] = logmask
                logscopes[:,slot:slot+1] = nls
            else:
                logmasks[:,slot:slot+1] = logscopes[:,slot:slot+1]
        return logmasks, logscopes


class MONet(BetaVAE):
    def __init__(self,
                 gamma=0.5,
                 input_shape=[3, 64, 64], 
                 nbr_attention_slot=10,
                 anet_basis_nbr_channel=32,
                 anet_block_depth=3,
                 cvae_beta=0.5,
                 cvae_latent_dim=10,
                 cvae_decoder_conv_dim=32, 
                 cvae_pretrained=False, 
                 cvae_resnet_encoder=False,
                 cvae_resnet_nbr_layer=2,
                 cvae_decoder_nbr_layer=3,
                 cvae_EncodingCapacityStep=None,
                 cvae_maxEncodingCapacity=100,
                 cvae_nbrEpochTillMaxEncodingCapacity=4,
                 cvae_constrainedEncoding=True,
                 cvae_observation_sigma=0.05,
                 compactness_factor=None):
        from skimage import segmentation
        cvae_input_shape = copy.deepcopy(input_shape)
        cvae_input_shape[0] += 1
        super(MONet, self).__init__(beta=cvae_beta, 
                                    latent_dim=cvae_latent_dim,
                                    nbr_attention_slot=None,
                                    input_shape=cvae_input_shape, 
                                    decoder_conv_dim=cvae_decoder_conv_dim, 
                                    decoder_nbr_layer=cvae_decoder_nbr_layer,
                                    pretrained=cvae_pretrained, 
                                    resnet_encoder=cvae_resnet_encoder,
                                    resnet_nbr_layer=cvae_resnet_nbr_layer,
                                    NormalOutputDistribution=True,
                                    EncodingCapacityStep=cvae_EncodingCapacityStep,
                                    maxEncodingCapacity=cvae_maxEncodingCapacity,
                                    nbrEpochTillMaxEncodingCapacity=cvae_nbrEpochTillMaxEncodingCapacity,
                                    constrainedEncoding=cvae_constrainedEncoding,
                                    observation_sigma=cvae_observation_sigma)

        # Set attribute 'attention_weights' where to store each slot's attention mask:
        setattr(self.encoder, 'attention_weights', list() )
        # Set attribute 'attention_reconstructions' where to store each slot's reconstruction:
        setattr(self.encoder, 'attention_reconstructions', list() )

        self.gamma = gamma
        self.cvae_input_shape = cvae_input_shape
        self.input_shape = input_shape

        self.attention_network = AttentionNetwork(input_shape=input_shape,
                                                  in_channel=input_shape[0],
                                                  attention_basis_nbr_channel=anet_basis_nbr_channel,
                                                  attention_block_depth=anet_block_depth)

        self.nbr_attention_slot = nbr_attention_slot

        self.compactness_factor = compactness_factor
        if self.compactness_factor is not None:
            self.use_compactness_constraint = True
        else:
            self.use_compactness_constraint = False

    def get_feature_shape(self):
        return self.latent_dim*self.nbr_attention_slot

    def encodeZ(self,x) :
        self.forward(x) 
        return self.z, self.mu, self.logvar

    def decode(self, z):
        batch_size = z.size(0)
        reconstructions = torch.empty(batch_size,
                                      self.nbr_attention_slot,
                                      *self.input_shape).to(z.device)
        mask_reconstructions = torch.empty(batch_size, 
                                           self.nbr_attention_slot, 
                                           1,
                                           *self.input_shape[-2:]).to(z.device)
        for slot, slot_z in zip(range(self.nbr_attention_slot), torch.chunk(z, self.nbr_attention_slot, dim=1)):
            cvae_out = self.decoder(slot_z)
            x_rec, log_mask_rec = torch.split(cvae_out, self.input_shape[0], dim=1)
            reconstructions[:,slot] = x_rec
            mask_reconstructions[:,slot] = torch.clamp_min(log_mask_rec.exp(), min=1e-9)

        reconstructions = torch.sum( mask_reconstructions * reconstructions, dim=1)
        return reconstructions

    def forward(self, 
                x,
                observation_sigma=None,
                compute_loss=False):
        if observation_sigma is None:
            observation_sigma = self.observation_sigma

        batch_size = x.size(0)

        initial_scope = torch.zeros(1, 1, *self.input_shape[-2:])
        log_scope = initial_scope.repeat(batch_size, 1 ,1, 1).to(x.device)

        logprobs = torch.empty(batch_size, 
                               self.nbr_attention_slot, 
                               *self.input_shape).to(x.device)
        self.reconstructions = torch.empty_like(logprobs)
        self.masks = torch.empty(batch_size, 
                                 self.nbr_attention_slot, 
                                 1,
                                 *self.input_shape[-2:]).to(x.device)
        self.log_mask_reconstructions = torch.empty_like(self.masks)

        scale = torch.empty((1,1,1,1)).fill_(self.observation_sigma).expand_as(x).to(x.device)

        per_slot_kls = list()
        self.mus = list()
        self.logvars = list()

        log_masks = self.attention_network(x=x, logscope=log_scope)
        for slot in range(self.nbr_attention_slot):
            if slot < self.nbr_attention_slot-1:
                log_mask, log_scope = self.attention_network(x=x, logscope=log_scope)
            else:
                log_mask = log_scope

            vae_in = torch.cat((x, log_mask), dim=1)
            mu, logvar, cvae_out = self._forward(vae_in)
            self.mus.append(mu)
            self.logvars.append(logvar)

            # Reconstructions Distributions:
            x_rec, log_mask_rec = torch.split(cvae_out, self.input_shape[0], dim=1)
            #rec_dist = Normal(x_rec, scale)
            if slot == 0:
                rec_dist = torch.distributions.Normal(x_rec, 0.9*scale)
            else:
                rec_dist = torch.distributions.Normal(x_rec, scale)

            logprobs[:, slot] = log_mask + rec_dist.log_prob(x)

            # KL divergence with latent prior:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            per_slot_kls.append(kl.unsqueeze(1))

            mask_probs = torch.clamp_min(log_mask.exp(), min=1e-9)
            
            self.masks[:, slot] = mask_probs
            self.reconstructions[:, slot] = x_rec
            self.log_mask_reconstructions[:, slot] = log_mask_rec

        per_slot_kls = torch.cat(per_slot_kls, dim=1)
        # batch_size x nbr_attention_slot x latent_dim

        self.mu = torch.cat(self.mus, dim=-1)
        self.logvar = torch.cat(self.logvars, dim=-1)
        self.z = self.reparameterize(self.mu, self.logvar)

        setattr(self.encoder, 'attention_weights', self.masks.data)
        setattr(self.encoder, 'attention_reconstructions', self.reconstructions.data)

        if not compute_loss:
            self.reconstruction = torch.sum( self.masks * self.reconstructions, dim=1)
            return self.reconstruction, self.mus, self.logvars, self.reconstructions, self.masks

        
        # sum over latent dimension and slots:
        kl_sum = per_slot_kls.sum(-1).sum(-1)
        # batch_size , 1

        # sum (exp O log) prob over slot dimension, 
        # compute log likelyhood and sum over input shape:
        neg_log_lik = -torch.logsumexp(logprobs, dim=1).view(batch_size,-1).sum(-1)
        # batch_size, 1

        # softmax over the slot dimension:
        self.log_mask_reconstructions = F.log_softmax(self.log_mask_reconstructions, dim=1)
        
        # softmax over the slot dimension:
        #self.maks = F.softmax(self.masks, dim=1)
        
        # mask reconstruction loss :: kl div loss: expects log likelihood input and probabilities targets:
        mask_reconstruction_loss = F.kl_div(input=self.log_mask_reconstructions, 
                                            target=self.masks,
                                            reduction='none').view(batch_size, -1).sum(-1)
        # batch_size, 1

        if self.use_compactness_constraint:
            #
            # Adapted from: 
            # https://github.com/kanezaki/pytorch-unsupervised-segmentation/blob/master/demo.py
            #
            nx = x.cpu().numpy().astype('double')*255.
            nx = nx.transpose((0, 2, 3, 1))
            b_nx = [nx[i] for i in range(len(nx))]
            b_labels = [segmentation.slic(nx, compactness=100.0, n_segments=10000) for nx in b_nx]
            b_labels = [labels.reshape(self.input_shape[-1]**2) for labels in b_labels]#nx.shape[0]*nx.shape[1])
            u_b_labels = [np.unique(labels) for labels in b_labels]
            l_b_inds = []
            for idx_u, u_labels in enumerate(u_b_labels):
                l_inds = []
                for i in range(len(u_labels)):
                    l_inds.append( np.where( b_labels[idx_u] == u_b_labels[idx_u][ i ] )[ 0 ] )
                l_b_inds.append(l_inds)

            compactness_loss_fn = torch.nn.CrossEntropyLoss()

            slot_logits_ppx = self.masks.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, -1, self.nbr_attention_slot)
            # batch x dim**2 x nbr_attention_slots
            slot_probs_ppx = torch.softmax(slot_logits_ppx, dim=-1)
            # batch x dim**2 x nbr_attention_slots
            _, target = torch.max( slot_probs_ppx, -1 )
            # batch x dim**2 x 1
            im_target = target.data.cpu().numpy()
            # batch x dim**2 x nbr_attention_slots

            im_vis = target[0].data.cpu().numpy()
            # dim**2 x 1
            nLabels = len(np.unique(im_vis))
            label_colours = np.random.randint(255,size=(100,3))
            im_vis_rgb = np.array([label_colours[ c % 100 ] for c in im_vis])
            im_vis_rgb = im_vis_rgb.reshape(*self.input_shape[-2:], 3).astype( np.uint8 )
            im_input_rgb = nx[0].reshape(*self.input_shape[-2:], 3).astype( np.uint8 )
            im_vis_rgb = np.concatenate([im_vis_rgb, im_input_rgb], axis=1)
            cv2.imwrite( "./mrl_rep_labels.png", im_vis_rgb )

            compactness_losses = []
            for idx_b in range(len(b_nx)):
                for i in range(len(l_inds)):
                    labels_per_sp = im_target[idx_b][ l_b_inds[idx_b][i] ]
                    u_labels_per_sp = np.unique( labels_per_sp )
                    hist = np.zeros( len(u_labels_per_sp) )
                    for j in range(len(hist)):
                        hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
                    im_target[idx_b][l_b_inds[idx_b][i]] = u_labels_per_sp[ np.argmax( hist ) ]
                target = torch.from_numpy(im_target[idx_b])
                target = target.to(x.device)
                compactness_losses.append(compactness_loss_fn(slot_logits_ppx[idx_b], target).sum().unsqueeze(0))
            compactness_losses = torch.cat(compactness_losses, dim=0)
            # batch x 1
            #
            #
        else:
            compactness_losses = None

        return self.reconstructions, self.masks, neg_log_lik, kl_sum, mask_reconstruction_loss, compactness_losses

    def compute_loss(self,
                     x=None,
                     observation_sigma=None):
        self.reconstructions, \
        self.mask_reconstructions, \
        self.neg_log_lik, \
        self.kl_sum, \
        self.mask_reconstruction_loss, \
        self.compactness_losses = self.forward(x=x,
                                              observation_sigma=observation_sigma,
                                              compute_loss=True)
        
        #--------------------------------------------------------------------------------------------------------------
        # Reconstruction loss :
        #self.neg_log_lik = -Normal(self.VAE_output, self.observation_sigma).log_prob( gtx)
        #self.reconst_loss = torch.sum( self.neg_log_lik.view( self.batch_size, -1), dim=1)
        #--------------------------------------------------------------------------------------------------------------
        # KL Divergence :
        self.true_kl_divergence = self.kl_sum #0.5 * (self.mu**2 + torch.exp(self.log_var) - self.log_var -1)
        self.kl_divergence_regularized = torch.zeros_like(self.true_kl_divergence)
        #--------------------------------------------------------------------------------------------------------------
        # VAE Loss:
        #--------------------------------------------------------------------------------------------------------------
        if self.EncodingCapacityStep is None :
            self.VAE_loss = self.neg_log_lik + self.beta*self.true_kl_divergence
        else:
            self.kl_divergence_regularized =  torch.abs( self.true_kl_divergence - self.EncodingCapacity )
            self.VAE_loss = self.neg_log_lik + self.beta*self.kl_divergence_regularized
            
            if self.increaseEncodingCapacity and self.training:
                self.EncodingCapacity += self.EncodingCapacityStep
            if self.EncodingCapacity >= self.maxEncodingCapacity :
                self.increaseEncodingCapacity = False 
        #--------------------------------------------------------------------------------------------------------------
        # MONet Loss:
        #--------------------------------------------------------------------------------------------------------------
        self.MONet_loss = self.VAE_loss + self.gamma*self.mask_reconstruction_loss
        #--------------------------------------------------------------------------------------------------------------
        
        if self.compactness_losses is not None:
            self.MONet_loss += self.compactness_factor*self.compactness_losses

        return self.MONet_loss, self.neg_log_lik, self.kl_divergence_regularized, self.true_kl_divergence


class ParallelMONet(BetaVAE):
    def __init__(self,
                 gamma=0.5,
                 input_shape=[3, 64, 64], 
                 nbr_attention_slot=10,
                 anet_basis_nbr_channel=32,
                 anet_block_depth=3,
                 cvae_beta=0.5,
                 cvae_latent_dim=10,
                 cvae_decoder_conv_dim=32, 
                 cvae_pretrained=False, 
                 cvae_resnet_encoder=False,
                 cvae_resnet_nbr_layer=2,
                 cvae_decoder_nbr_layer=3,
                 cvae_EncodingCapacityStep=None,
                 cvae_maxEncodingCapacity=100,
                 cvae_nbrEpochTillMaxEncodingCapacity=4,
                 cvae_constrainedEncoding=True,
                 cvae_observation_sigma=0.05,
                 compactness_factor=None):
        from skimage import segmentation
        cvae_input_shape = copy.deepcopy(input_shape)
        cvae_input_shape[0] += 1
        super(ParallelMONet, self).__init__(beta=cvae_beta, 
                                            latent_dim=cvae_latent_dim,
                                            nbr_attention_slot=None,
                                            input_shape=cvae_input_shape, 
                                            decoder_conv_dim=cvae_decoder_conv_dim, 
                                            decoder_nbr_layer=cvae_decoder_nbr_layer,
                                            pretrained=cvae_pretrained, 
                                            resnet_encoder=cvae_resnet_encoder,
                                            resnet_nbr_layer=cvae_resnet_nbr_layer,
                                            NormalOutputDistribution=True,
                                            EncodingCapacityStep=cvae_EncodingCapacityStep,
                                            maxEncodingCapacity=cvae_maxEncodingCapacity,
                                            nbrEpochTillMaxEncodingCapacity=cvae_nbrEpochTillMaxEncodingCapacity,
                                            constrainedEncoding=cvae_constrainedEncoding,
                                            observation_sigma=cvae_observation_sigma)

        # Set attribute 'attention_weights' where to store each slot's attention mask:
        setattr(self.encoder, 'attention_weights', list() )
        # Set attribute 'attention_reconstructions' where to store each slot's reconstruction:
        setattr(self.encoder, 'attention_reconstructions', list() )

        self.gamma = gamma
        self.cvae_input_shape = cvae_input_shape
        self.input_shape = input_shape
        self.nbr_attention_slot = nbr_attention_slot

        self.attention_network = ParallelAttentionNetwork(input_shape=input_shape,
                                                          in_channel=input_shape[0],
                                                          nbr_attention_slot=self.nbr_attention_slot,
                                                          attention_basis_nbr_channel=anet_basis_nbr_channel,
                                                          attention_block_depth=anet_block_depth)


        self.compactness_factor = compactness_factor
        if self.compactness_factor is not None:
            self.use_compactness_constraint = True
        else:
            self.use_compactness_constraint = False

    def get_feature_shape(self):
        return self.latent_dim*self.nbr_attention_slot

    def encodeZ(self,x) :
        self.forward(x) 
        return self.z, self.mu, self.logvar

    def decode(self, z):
        batch_size = z.size(0)
        reconstructions = torch.empty(batch_size,
                                      self.nbr_attention_slot,
                                      *self.input_shape).to(z.device)
        mask_reconstructions = torch.empty(batch_size, 
                                           self.nbr_attention_slot, 
                                           1,
                                           *self.input_shape[-2:]).to(z.device)
        for slot, slot_z in zip(range(self.nbr_attention_slot), torch.chunk(z, self.nbr_attention_slot, dim=1)):
            cvae_out = self.decoder(slot_z)
            x_rec, log_mask_rec = torch.split(cvae_out, self.input_shape[0], dim=1)
            reconstructions[:,slot] = x_rec
            mask_reconstructions[:,slot] = torch.clamp_min(log_mask_rec.exp(), min=1e-9)

        reconstructions = torch.sum( mask_reconstructions * reconstructions, dim=1)
        return reconstructions

    def forward(self, 
                x,
                observation_sigma=None,
                compute_loss=False):
        if observation_sigma is None:
            observation_sigma = self.observation_sigma

        batch_size = x.size(0)

        initial_scope = torch.zeros(1, 1, *self.input_shape[-2:])
        log_scope = initial_scope.repeat(batch_size, 1 ,1, 1).to(x.device)

        logprobs = torch.empty(batch_size, 
                               self.nbr_attention_slot, 
                               *self.input_shape).to(x.device)
        self.reconstructions = torch.empty_like(logprobs)
        self.masks = torch.empty(batch_size, 
                                 self.nbr_attention_slot, 
                                 1,
                                 *self.input_shape[-2:]).to(x.device)
        self.log_mask_reconstructions = torch.empty_like(self.masks)

        scale = torch.empty((1,1,1,1)).fill_(self.observation_sigma).expand_as(x).to(x.device)

        per_slot_kls = list()
        self.mus = list()
        self.logvars = list()

        log_masks, log_scopes = self.attention_network(x=x, logscope=log_scope)
        for slot in range(self.nbr_attention_slot):
            log_mask = log_masks[:,slot:slot+1]

            vae_in = torch.cat((x, log_mask), dim=1)
            mu, logvar, cvae_out = self._forward(vae_in)
            self.mus.append(mu)
            self.logvars.append(logvar)

            # Reconstructions Distributions:
            x_rec, log_mask_rec = torch.split(cvae_out, self.input_shape[0], dim=1)
            #rec_dist = Normal(x_rec, scale)
            if slot == 0:
                rec_dist = torch.distributions.Normal(x_rec, 0.9*scale)
            else:
                rec_dist = torch.distributions.Normal(x_rec, scale)

            logprobs[:, slot] = log_mask + rec_dist.log_prob(x)

            # KL divergence with latent prior:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            per_slot_kls.append(kl.unsqueeze(1))

            mask_probs = torch.clamp_min(log_mask.exp(), min=1e-9)
            
            self.masks[:, slot] = mask_probs
            self.reconstructions[:, slot] = x_rec
            self.log_mask_reconstructions[:, slot] = log_mask_rec

        per_slot_kls = torch.cat(per_slot_kls, dim=1)
        # batch_size x nbr_attention_slot x latent_dim

        self.mu = torch.cat(self.mus, dim=-1)
        self.logvar = torch.cat(self.logvars, dim=-1)
        self.z = self.reparameterize(self.mu, self.logvar)

        setattr(self.encoder, 'attention_weights', self.masks.data)
        setattr(self.encoder, 'attention_reconstructions', self.reconstructions.data)

        if not compute_loss:
            self.reconstruction = torch.sum( self.masks * self.reconstructions, dim=1)
            return self.reconstruction, self.mus, self.logvars, self.reconstructions, self.masks

        
        # sum over latent dimension and slots:
        kl_sum = per_slot_kls.sum(-1).sum(-1)
        # batch_size , 1

        # sum (exp O log) prob over slot dimension, 
        # compute log likelyhood and sum over input shape:
        neg_log_lik = -torch.logsumexp(logprobs, dim=1).view(batch_size,-1).sum(-1)
        # batch_size, 1

        # softmax over the slot dimension:
        self.log_mask_reconstructions = F.log_softmax(self.log_mask_reconstructions, dim=1)
        
        # softmax over the slot dimension:
        #self.maks = F.softmax(self.masks, dim=1)
        
        # mask reconstruction loss :: kl div loss: expects log likelihood input and probabilities targets:
        mask_reconstruction_loss = F.kl_div(input=self.log_mask_reconstructions, 
                                            target=self.masks,
                                            reduction='none').view(batch_size, -1).sum(-1)
        # batch_size, 1

        if self.use_compactness_constraint:
            #
            # Adapted from: 
            # https://github.com/kanezaki/pytorch-unsupervised-segmentation/blob/master/demo.py
            #
            nx = x.cpu().numpy().astype('double')*255.
            nx = nx.transpose((0, 2, 3, 1))
            b_nx = [nx[i] for i in range(len(nx))]
            b_labels = [segmentation.slic(nx, compactness=100.0, n_segments=10000) for nx in b_nx]
            b_labels = [labels.reshape(self.input_shape[-1]**2) for labels in b_labels]#nx.shape[0]*nx.shape[1])
            u_b_labels = [np.unique(labels) for labels in b_labels]
            l_b_inds = []
            for idx_u, u_labels in enumerate(u_b_labels):
                l_inds = []
                for i in range(len(u_labels)):
                    l_inds.append( np.where( b_labels[idx_u] == u_b_labels[idx_u][ i ] )[ 0 ] )
                l_b_inds.append(l_inds)

            compactness_loss_fn = torch.nn.CrossEntropyLoss()

            slot_logits_ppx = self.masks.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, -1, self.nbr_attention_slot)
            # batch x dim**2 x nbr_attention_slots
            slot_probs_ppx = torch.softmax(slot_logits_ppx, dim=-1)
            # batch x dim**2 x nbr_attention_slots
            _, target = torch.max( slot_probs_ppx, -1 )
            # batch x dim**2 x 1
            im_target = target.data.cpu().numpy()
            # batch x dim**2 x nbr_attention_slots

            im_vis = target[0].data.cpu().numpy()
            # dim**2 x 1
            nLabels = len(np.unique(im_vis))
            label_colours = np.random.randint(255,size=(100,3))
            im_vis_rgb = np.array([label_colours[ c % 100 ] for c in im_vis])
            im_vis_rgb = im_vis_rgb.reshape(*self.input_shape[-2:], 3).astype( np.uint8 )
            im_input_rgb = nx[0].reshape(*self.input_shape[-2:], 3).astype( np.uint8 )
            im_vis_rgb = np.concatenate([im_vis_rgb, im_input_rgb], axis=1)
            cv2.imwrite( "./mrl_rep_parallel_labels.png", im_vis_rgb )

            compactness_losses = []
            for idx_b in range(len(b_nx)):
                for i in range(len(l_inds)):
                    labels_per_sp = im_target[idx_b][ l_b_inds[idx_b][i] ]
                    u_labels_per_sp = np.unique( labels_per_sp )
                    hist = np.zeros( len(u_labels_per_sp) )
                    for j in range(len(hist)):
                        hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
                    im_target[idx_b][l_b_inds[idx_b][i]] = u_labels_per_sp[ np.argmax( hist ) ]
                target = torch.from_numpy(im_target[idx_b])
                target = target.to(x.device)
                compactness_losses.append(compactness_loss_fn(slot_logits_ppx[idx_b], target).sum().unsqueeze(0))
            compactness_losses = torch.cat(compactness_losses, dim=0)
            # batch x 1
            #
            #
        else:
            compactness_losses = None

        return self.reconstructions, self.masks, neg_log_lik, kl_sum, mask_reconstruction_loss, compactness_losses

    def compute_loss(self,
                     x=None,
                     observation_sigma=None):
        self.reconstructions, \
        self.mask_reconstructions, \
        self.neg_log_lik, \
        self.kl_sum, \
        self.mask_reconstruction_loss, \
        self.compactness_losses = self.forward(x=x,
                                              observation_sigma=observation_sigma,
                                              compute_loss=True)
        
        #--------------------------------------------------------------------------------------------------------------
        # Reconstruction loss :
        #self.neg_log_lik = -Normal(self.VAE_output, self.observation_sigma).log_prob( gtx)
        #self.reconst_loss = torch.sum( self.neg_log_lik.view( self.batch_size, -1), dim=1)
        #--------------------------------------------------------------------------------------------------------------
        # KL Divergence :
        self.true_kl_divergence = self.kl_sum #0.5 * (self.mu**2 + torch.exp(self.log_var) - self.log_var -1)
        self.kl_divergence_regularized = torch.zeros_like(self.true_kl_divergence)
        #--------------------------------------------------------------------------------------------------------------
        # VAE Loss:
        #--------------------------------------------------------------------------------------------------------------
        if self.EncodingCapacityStep is None :
            self.VAE_loss = self.neg_log_lik + self.beta*self.true_kl_divergence
        else:
            self.kl_divergence_regularized =  torch.abs( self.true_kl_divergence - self.EncodingCapacity )
            self.VAE_loss = self.neg_log_lik + self.beta*self.kl_divergence_regularized
            
            if self.increaseEncodingCapacity and self.training:
                self.EncodingCapacity += self.EncodingCapacityStep
            if self.EncodingCapacity >= self.maxEncodingCapacity :
                self.increaseEncodingCapacity = False 
        #--------------------------------------------------------------------------------------------------------------
        # MONet Loss:
        #--------------------------------------------------------------------------------------------------------------
        self.MONet_loss = self.VAE_loss + self.gamma*self.mask_reconstruction_loss
        #--------------------------------------------------------------------------------------------------------------
        
        if self.compactness_losses is not None:
            self.MONet_loss += self.compactness_factor*self.compactness_losses

        return self.MONet_loss, self.neg_log_lik, self.kl_divergence_regularized, self.true_kl_divergence


