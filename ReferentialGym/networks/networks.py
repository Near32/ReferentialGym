import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision import models
from torchvision.models.resnet import model_urls, BasicBlock



def hasnan(tensor):
    if torch.isnan(tensor).max().item() == 1:
        return True
    return False

def handle_nan(layer, verbose=True):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        nan_indices = torch.isnan(layer._parameters[name].data)
        if verbose and torch.any(nan_indices).item(): print("WARNING: NaN found in {}.".format(name))
        layer._parameters[name].data[nan_indices]=0
        if param.grad is None: continue
        nan_indices = torch.isnan(layer._parameters[name].grad.data)
        if verbose and torch.any(nan_indices).item(): print("WARNING: NaN found in the GRADIENT of {}.".format(name))
        layer._parameters[name].grad.data[nan_indices]=0
        
def layer_init(layer, w_scale=1.0):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        if 'bias' in name:
            #layer._parameters[name].data.fill_(0.0)
            layer._parameters[name].data.uniform_(-0.08,0.08)
        else:
            #nn.init.orthogonal_(layer._parameters[name].data)
            '''
            fanIn = param.size(0)
            fanOut = param.size(1)

            factor = math.sqrt(2.0/(fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            layer._parameters[name].data.copy_(weight)
            '''
            
            '''
            layer._parameters[name].data.uniform_(-0.08,0.08)
            layer._parameters[name].data.mul_(w_scale)
            '''
            nn.init.kaiming_normal_(layer._parameters[name], mode="fan_out", nonlinearity='leaky_relu')
            
    '''
    if hasattr(layer,"weight"):    
        #nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight.data.mul_(w_scale)
        if hasattr(layer,"bias") and layer.bias is not None:    
            #nn.init.constant_(layer.bias.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
        
    if hasattr(layer,"weight_ih"):
        #nn.init.orthogonal_(layer.weight_ih.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight_ih.data.mul_(w_scale)
        if hasattr(layer,"bias_ih"):    
            #nn.init.constant_(layer.bias_ih.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
        
    if hasattr(layer,"weight_hh"):    
        #nn.init.orthogonal_(layer.weight_hh.data)
        layer.weight.data.uniform_(-0.08,0.08)
        layer.weight_hh.data.mul_(w_scale)
        if hasattr(layer,"bias_hh"):    
            #nn.init.constant_(layer.bias_hh.data, 0)
            layer.bias.data.uniform_(-0.08,0.08)
    '''

    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = self.gate(x)
        return x

    def get_feature_shape(self):
        return self.feature_dim


class ConvolutionalBody(nn.Module):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3], kernel_sizes=[1], strides=[1], paddings=[0], non_linearities=[F.leaky_relu]):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalBody, self).__init__()
        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities] * (len(channels) - 1)
        else:
            while len(self.non_linearities) <= (len(channels) - 1):
                self.non_linearities.append(self.non_linearities[0])

        self.feature_dim = feature_dim
        if isinstance(feature_dim, tuple):
            self.feature_dim = feature_dim[-1]

        self.convs = nn.ModuleList()
        dim = input_shape[1] # height
        for in_ch, out_ch, k, s, p in zip(channels, channels[1:], kernel_sizes, strides, paddings):
            self.convs.append( layer_init(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p), w_scale=math.sqrt(2)))
            # Update of the shape of the input-image, following Conv:
            dim = (dim-k+2*p)//s+1
            print(dim)
            
        hidden_units = (dim * dim * channels[-1],)
        if isinstance(feature_dim, tuple):
            hidden_units = hidden_units + feature_dim
        else:
            hidden_units = hidden_units + (self.feature_dim,)

        self.fcs = nn.ModuleList()
        for nbr_in, nbr_out in zip(hidden_units, hidden_units[1:]):
            self.fcs.append( layer_init(nn.Linear(nbr_in, nbr_out), w_scale=math.sqrt(2)))#1e-2))#1.0/math.sqrt(nbr_in*nbr_out)))

    def forward(self, x, non_lin_output=True):
        conv_map = x
        for conv_layer, non_lin in zip(self.convs, self.non_linearities):
            conv_map = non_lin(conv_layer(conv_map))

        features = conv_map.view(conv_map.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            features = fc(features)
            if idx != len(self.fcs)-1 or non_lin_output:
                features = F.relu(features)

        return features

    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.feature_dim


class ConvolutionalLstmBody(ConvolutionalBody):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3], kernel_sizes=[1], strides=[1], paddings=[0], non_linearities=[F.relu], hidden_units=(256,), gate=F.relu):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalLstmBody, self).__init__(input_shape=input_shape,
                                                feature_dim=feature_dim,
                                                channels=channels,
                                                kernel_sizes=kernel_sizes,
                                                strides=strides,
                                                paddings=paddings,
                                                non_linearities=non_linearities)

        self.lstm_body = LSTMBody( state_dim=self.feature_dim, hidden_units=hidden_units, gate=gate)

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        features = super(ConvolutionalLstmBody,self).forward(x)
        return self.lstm_body( (features, recurrent_neurons))

    def get_reset_states(self, cuda=False, repeat=1):
        return self.lstm_body.get_reset_states(cuda=cuda, repeat=repeat)
    
    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.lstm_body.get_feature_shape()


class ConvolutionalGruBody(ConvolutionalBody):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3], kernel_sizes=[1], strides=[1], paddings=[0], non_linearities=[F.relu], hidden_units=(256,), gate=F.relu):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalGruBody, self).__init__(input_shape=input_shape,
                                                feature_dim=feature_dim,
                                                channels=channels,
                                                kernel_sizes=kernel_sizes,
                                                strides=strides,
                                                paddings=paddings,
                                                non_linearities=non_linearities)

        self.gru_body = GRUBody( state_dim=self.feature_dim, hidden_units=hidden_units, gate=gate)

    def forward(self, inputs):
        '''
        :param inputs: input to GRU cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        features = super(ConvolutionalGruBody,self).forward(x)
        return self.gru_body( (features, recurrent_neurons))

    def get_reset_states(self, cuda=False, repeat=1):
        return self.gru_body.get_reset_states(cuda=cuda, repeat=repeat)

    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.gru_body.get_feature_shape()


class LSTMBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256), gate=F.relu):
        super(LSTMBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList([layer_init(nn.LSTMCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.gate = gate

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']

        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            nhx, ncx = layer(x, (hx, cx))
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            # Consider not applying activation functions on last layer's output
            if self.gate is not None:
                x = self.gate(nhx)

        return x, {'hidden': next_hstates, 'cell': next_cstates}

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return {'hidden': hidden_states, 'cell': cell_states}

    def get_feature_shape(self):
        return self.feature_dim


class GRUBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256), gate=F.relu):
        super(GRUBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList([layer_init(nn.GRUCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.gate = gate

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']

        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            nhx = layer(x, hx)
            next_hstates.append(nhx)
            next_cstates.append(nhx)
            # Consider not applying activation functions on last layer's output
            if self.gate is not None:
                x = self.gate(nhx)

        return x, {'hidden': next_hstates, 'cell': next_cstates}

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return {'hidden': hidden_states, 'cell': cell_states}

    def get_feature_shape(self):
        return self.feature_dim

class ModelResNet18(models.ResNet) :
    def __init__(self, input_shape, feature_dim=256, nbr_layer=None, pretrained=False, **kwargs):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param nbr_layer: int, number of convolutional residual layer to use.
        :param pretrained: bool, specifies whether to load a pretrained model.
        '''
        super(ModelResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
        self.input_shape = input_shape
        self.nbr_layer = nbr_layer
        
        # Re-organize the input conv layer:
        saved_kernel = self.conv1.weight.data
        
        if input_shape[0] >3:
            '''
            in3depth = input_shape[0] // 3
            concat_kernel = []
            for i in range(in3depth) :
                concat_kernel.append( saved_kernel)
            concat_kernel = torch.cat(concat_kernel, dim=1)

            self.conv1 = nn.Conv2d(in3depth*3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data = concat_kernel
            '''
            self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data[:,0:3,...] = saved_kernel
            
        elif input_shape[0] <3:
            self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data = saved_kernel[:,0:input_shape[0],...]

        # 64:
        self.avgpool_ksize = 2
        # 224:
        #self.avgpool_ksize = 7
        self.avgpool = nn.AvgPool2d(self.avgpool_ksize, stride=1)
        
        # Add the fully-connected layers at the top:
        self.feature_dim = feature_dim
        if isinstance(feature_dim, tuple):
            self.feature_dim = feature_dim[-1]

        # Compute the number of features:
        num_ftrs = self._compute_feature_dim(input_shape[-1], self.nbr_layer)

        self.fc = layer_init(nn.Linear(num_ftrs, self.feature_dim), w_scale=math.sqrt(2))
    
    def _compute_feature_dim(self, input_dim, nbr_layer):
        if nbr_layer is None: return self.fc.in_features

        layers_depths = [64,128,256,512]
        layers_divisions = [1,2,2,2]

        # Conv1:
        dim = input_dim // 2
        # MaxPool1:
        dim = dim // 2

        depth = 64
        for idx_layer in range(nbr_layer):
            dim = dim // layers_divisions[idx_layer]
            depth = layers_depths[idx_layer]

        # Avg Pool:
        dim -= 1

        return depth * dim * dim  

    def forward(self, x):
        #xsize = x.size()
        #print('input:',xsize)
        x = self.conv1(x)
        #xsize = x.size()
        #print('cv0:',xsize)
        x = self.bn1(x)
        x = self.relu(x)
        
        self.x0 = self.maxpool(x)
        
        #xsize = self.x0.size()
        #print('mxp0:',xsize)

        if self.nbr_layer >= 1 :
            self.x1 = self.layer1(self.x0)
            #xsize = self.x1.size()
            #print('1:',xsize)
            if self.nbr_layer >= 2 :
                self.x2 = self.layer2(self.x1)
                #xsize = self.x2.size()
                #print('2:',xsize)
                if self.nbr_layer >= 3 :
                    self.x3 = self.layer3(self.x2)
                    #xsize = self.x3.size()
                    #print('3:',xsize)
                    if self.nbr_layer >= 4 :
                        self.x4 = self.layer4(self.x3)
                        #xsize = self.x4.size()
                        #print('4:',xsize)
                        
                        self.features_map = self.x4
                    else :
                        self.features_map = self.x3
                else :
                    self.features_map = self.x2
            else :
                self.features_map = self.x1
        else :
            self.features_map = self.x0
        
        avgx = self.avgpool(self.features_map)
        #xsize = avgx.size()
        #print('avg : x :',xsize)
        fcx = avgx.view(avgx.size(0), -1)
        #xsize = fcx.size()
        #print('reg avg : x :',xsize)
        fcx = self.fc(fcx)
        #xsize = fcx.size()
        #print('fc output : x :',xsize)
        
        return fcx

    def get_feature_shape(self):
        return self.feature_dim
