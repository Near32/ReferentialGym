import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import layer_init, layer_init_lstm, layer_init_gru


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
        self.layers = nn.ModuleList([layer_init_lstm(nn.LSTMCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
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
        self.layers = nn.ModuleList([layer_init_gru(nn.GRUCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
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
        