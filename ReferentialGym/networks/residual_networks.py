import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision import models
from torchvision.models.resnet import model_urls, BasicBlock

from .networks import layer_init

from .networks import coordconv, coorddeconv, conv1x1, coordconv1x1, conv3x3, coordconv3x3

# From torchvision.models.resnet:
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class CoordResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        nn.Module.__init__(self)
        
        self.addXY_initial = True 

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = coordconv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        inplanes = self.inplanes
        if self.addXY_initial:   
            inplanes +=2 
            self.addXY_initial = False
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        '''
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        '''
        layers.append(block(inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ModelResNet18(models.ResNet):
    def __init__(self, input_shape, feature_dim=256, nbr_layer=None, pretrained=False, use_coordconv=False):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param nbr_layer: int, number of convolutional residual layer to use.
        :param pretrained: bool, specifies whether to load a pretrained model.
        '''
        '''
        if use_coordconv:
            models.resnet.ResNet = CoordResNet
        super(ModelResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        if use_coordconv:
            models.resnet.ResNet = ResNet
        '''
        super_class = ResNet
        if use_coordconv:
            super_class = CoordResNet
        super_class.__init__(self, BasicBlock, [2, 2, 2, 2])
        
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
        self.input_shape = input_shape
        self.nbr_layer = nbr_layer
        
        # Re-organize the input conv layer:
        saved_kernel = self.conv1.weight.data
        
        conv_fn = nn.Conv2d
        if use_coordconv:   conv_fn = coordconv
        if input_shape[0] >3:
            self.conv1 = conv_fn(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1.weight.data[:,0:3,...] = saved_kernel
            
        elif input_shape[0] <3:
            self.conv1 = conv_fn(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.feat_map_dim, self.feat_map_depth = self._compute_feature_shape(input_shape[-1], self.nbr_layer)
        # Avg Pool:
        feat_dim = self.feat_map_dim-1
        num_ftrs = self.feat_map_depth * feat_dim * feat_dim
        
        self.fc = layer_init(nn.Linear(num_ftrs, self.feature_dim), w_scale=math.sqrt(2))
    
    def _compute_feature_shape(self, input_dim, nbr_layer):
        if nbr_layer is None: return self.fc.in_features

        layers_depths = [64,128,256,512]
        layers_divisions = [1,2,2,2]

        # Conv1:
        dim = input_dim // 2
        # MaxPool1:
        dim = dim // 2

        depth = 64
        for idx_layer in range(nbr_layer):
            dim = math.ceil(float(dim) / layers_divisions[idx_layer])
            depth = layers_depths[idx_layer]
            print(dim, depth)

        return dim, depth

    def _compute_feat_map(self, x):
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

        if self.nbr_layer >= 1:
            self.x1 = self.layer1(self.x0)
            #xsize = self.x1.size()
            #print('1:',xsize)
            if self.nbr_layer >= 2:
                self.x2 = self.layer2(self.x1)
                #xsize = self.x2.size()
                #print('2:',xsize)
                if self.nbr_layer >= 3:
                    self.x3 = self.layer3(self.x2)
                    #xsize = self.x3.size()
                    #print('3:',xsize)
                    if self.nbr_layer >= 4:
                        self.x4 = self.layer4(self.x3)
                        #xsize = self.x4.size()
                        #print('4:',xsize)
                        
                        self.features_map = self.x4
                    else:
                        self.features_map = self.x3
                else:
                    self.features_map = self.x2
            else:
                self.features_map = self.x1
        else:
            self.features_map = self.x0
        
        return self.features_map

    def _compute_features(self, features_map):
        avgx = self.avgpool(features_map)
        #xsize = avgx.size()
        #print('avg: x:',xsize)
        fcx = avgx.view(avgx.size(0), -1)
        #xsize = fcx.size()
        #print('reg avg: x:',xsize)
        fcx = self.fc(fcx)
        #xsize = fcx.size()
        #print('fc output: x:',xsize)
        return fcx

    def forward(self, x):
        self.features_map = self._compute_feat_map(x)
        self.features = self._compute_features(self.features_map)
        return self.features

    def get_feature_shape(self):
        return self.feature_dim


class ModelResNet18AvgPooled(models.ResNet):
    def __init__(self, input_shape, feature_dim=256, nbr_layer=None, pretrained=False, detach_conv_maps=False, use_coordconv=False):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param nbr_layer: int, number of convolutional residual layer to use.
        :param pretrained: bool, specifies whether to load a pretrained model.
        '''
        '''
        if use_coordconv:
            models.resnet.ResNet = CoordResNet
        super(ModelResNet18AvgPooled, self).__init__(BasicBlock, [2, 2, 2, 2])
        if use_coordconv:
            models.resnet.ResNet = ResNet
        '''
        super_class = ResNet
        if use_coordconv:
            super_class = CoordResNet
        super_class.__init__(self, BasicBlock, [2, 2, 2, 2])
        
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
        self.input_shape = input_shape
        self.nbr_layer = nbr_layer
        self.detach_conv_maps = detach_conv_maps
        
        # Re-organize the input conv layer:
        if pretrained:
            if use_coordconv:
                saved_kernel = self.conv1[0].weight.data
            else:
                saved_kernel = self.conv1.weight.data
        
        conv_fn = nn.Conv2d
        if use_coordconv:   conv_fn = coordconv
        if input_shape[0] >3:
            self.conv1 = conv_fn(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                if use_coordconv:
                    self.conv1[0].weight.data[:,0:3,...] = saved_kernel
                else:
                    self.conv1.weight.data[:,0:3,...] = saved_kernel            
        elif input_shape[0] <3:
            self.conv1 = conv_fn(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                if use_coordconv:
                    self.conv1[0].weight.data = saved_kernel[:,0:input_shape[0],...]
                else:
                    self.conv1.weight.data = saved_kernel[:,0:input_shape[0],...]

        # Compute the number of features:
        self.feat_map_dim, self.feat_map_depth = self._compute_feature_shape(input_shape[-1], self.nbr_layer)
        self.avgpool_ksize = self.feat_map_dim
        self.avgpool = nn.AvgPool2d(self.avgpool_ksize, stride=1, padding=0)
        
        # Avg Pool:
        num_ftrs = self.feat_map_depth
        
        # Add the fully-connected layers at the top:
        self.feature_dim = feature_dim
        if isinstance(feature_dim, tuple):
            self.feature_dim = feature_dim[-1]

        self.fc = layer_init(nn.Linear(num_ftrs, self.feature_dim), w_scale=math.sqrt(2))
    
    def _compute_feature_shape(self, input_dim=None, nbr_layer=None):
        if input_dim is None: input_dim = self.input_shape[-1]
        #if nbr_layer is None: return self.fc.in_features
        if nbr_layer is None: nbr_layer = self.nbr_layer

        layers_depths = [64,128,256,512]
        layers_divisions = [1,2,2,2]

        # Conv1:
        dim = input_dim // 2
        # MaxPool1:
        dim = dim // 2

        depth = 64
        for idx_layer in range(nbr_layer):
            dim = math.ceil(float(dim) / layers_divisions[idx_layer])
            depth = layers_depths[idx_layer]
            print(dim, depth)

        return dim, depth

    def _compute_feat_map(self, x):
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

        if self.nbr_layer >= 1:
            self.x1 = self.layer1(self.x0)
            #xsize = self.x1.size()
            #print('1:',xsize)
            if self.nbr_layer >= 2:
                self.x2 = self.layer2(self.x1)
                #xsize = self.x2.size()
                #print('2:',xsize)
                if self.nbr_layer >= 3:
                    self.x3 = self.layer3(self.x2)
                    #xsize = self.x3.size()
                    #print('3:',xsize)
                    if self.nbr_layer >= 4:
                        self.x4 = self.layer4(self.x3)
                        #xsize = self.x4.size()
                        #print('4:',xsize)
                        
                        self.features_map = self.x4
                    else:
                        self.features_map = self.x3
                else:
                    self.features_map = self.x2
            else:
                self.features_map = self.x1
        else:
            self.features_map = self.x0
        
        return self.features_map

    def _compute_features(self, features_map):
        avgx = self.avgpool(features_map)
        #xsize = avgx.size()
        #print('avg: x:',xsize)
        fcx = avgx.view(avgx.size(0), -1)
        #xsize = fcx.size()
        #print('reg avg: x:',xsize)
        fcx = self.fc(fcx)
        #xsize = fcx.size()
        #print('fc output: x:',xsize)
        return fcx

    def forward(self, x):
        self.features_map = self._compute_feat_map(x)
        self.features_comp_input = self.features_map.clone()
        if self.detach_conv_maps:   self.features_comp_input = self.features_comp_input.detach()
        self.features = self._compute_features(self.features_comp_input)
        return self.features

    def get_feat_map(self):
        return self.features_map

    def get_conv_map_shpae(self):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return shape
    def get_feature_shape(self):
        return self.feature_dim


from .networks import MHDPA_RN

class ResNet18MHDPA(ModelResNet18):
    def __init__(self, 
                 input_shape, 
                 feature_dim=256, 
                 nbr_layer=None, 
                 pretrained=False, 
                 use_coordconv=False,
                 dropout=0.0, 
                 non_linearities=[nn.LeakyReLU],
                 nbrHead=4,
                 nbrRecurrentSharedLayers=1,  
                 units_per_MLP_layer=512,
                 interaction_dim=128):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param nbr_layer: int, number of convolutional residual layer to use.
        :param pretrained: bool, specifies whether to load a pretrained model.
        :param dropout: dropout probability to use.
        :param non_linearities: list of non-linear nn.Functional functions to use
                                after each convolutional layer.
        :param nbrHead: Int, number of Scaled Dot-Product Attention head.
        :param nbrRecurrentSharedLayers: Int, number of recurrent update to apply.
        :param units_per_MLP_layer: Int, number of neurons in the transformation from the
                                    concatenated head outputs to the entity embedding space.
        :param interaction_dim: Int, number of dimensions in the interaction space.
        '''
        super(ResNet18MHDPA, self).__init__(input_shape=input_shape,
                                            feature_dim=feature_dim,
                                            nbr_layer=nbr_layer,
                                            pretrained=pretrained,
                                            use_coordconv=use_coordconv)       
        self.dropout = dropout
        self.relationModule = MHDPA_RN(output_dim=None,
                                       depth_dim=self.feat_map_depth+2,
                                       nbrHead=nbrHead, 
                                       nbrRecurrentSharedLayers=nbrRecurrentSharedLayers, 
                                       nbrEntity=self.feat_map_dim*self.feat_map_dim,  
                                       units_per_MLP_layer=units_per_MLP_layer,
                                       interactions_dim=interaction_dim,
                                       dropout_prob=self.dropout)
        
        hidden_units = (self.feat_map_dim * self.feat_map_dim * (self.feat_map_depth+2),)
        if isinstance(feature_dim, tuple):
            hidden_units = hidden_units + feature_dim
        else:
            hidden_units = hidden_units + (self.feature_dim,)

        self.fcs = nn.ModuleList()
        for nbr_in, nbr_out in zip(hidden_units, hidden_units[1:]):
            self.fcs.append( layer_init(nn.Linear(nbr_in, nbr_out), w_scale=math.sqrt(2)))#1e-2))#1.0/math.sqrt(nbr_in*nbr_out)))
            if self.dropout:
                self.fcs.append( nn.Dropout(p=self.dropout))

    def forward(self, x):
        x = self._compute_feat_map(x) 

        xsize = x.size()
        batchsize = xsize[0]
        depthsize = xsize[1]
        spatialSize = xsize[2]
        featuresize = spatialSize*spatialSize

        feat_map = self.relationModule(x)

        features = feat_map.view(feat_map.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            features = fc(features)
            features = F.leaky_relu(features)

        return features


class ExtractorResNet18(ModelResNet18):
    def __init__(self, input_shape, final_layer_idx=None, pretrained=True):
        super(ExtractorResNet18, self).__init__(input_shape=input_shape, feature_dim=1, nbr_layer=final_layer_idx, pretrained=pretrained)
        self.input_shape = input_shape
        self.final_layer_idx = final_layer_idx
        
    def forward(self, x):
        return self._compute_feat_map(x)
        