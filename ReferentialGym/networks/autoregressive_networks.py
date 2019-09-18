import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision import models
from torchvision.models.resnet import model_urls, BasicBlock

import math
from numbers import Number 

from .networks import ModelResNet18


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


class addXYfeatures(nn.Module) :
    def __init__(self) :
        super(addXYfeatures,self).__init__() 
        self.fXY = None

    def forward(self,x) :
        xsize = x.size()
        batch = xsize[0]
        if self.fXY is None:
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
            self.fXY = torch.cat( [fxy,fyx], dim=1)
        
        fXY = self.fXY.repeat(batch,1,1,1)
        if x.is_cuda : fXY = fXY.cuda()
            
        out = torch.cat( [x,fXY], dim=1)

        return out 

def conv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
    layers = []
    layers.append( nn.Conv2d( sin,sout, k, stride,pad,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )


def deconv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
    layers = []
    layers.append( nn.ConvTranspose2d( sin,sout, k, stride,pad,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coordconv( sin, sout,kernel_size,stride=2,pad=1,batchNorm=False,bias=False) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.Conv2d( sin+2,sout, kernel_size, stride,pad,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coorddeconv( sin, sout,kernel_size,stride=2,pad=1,batchNorm=True,bias=False) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.ConvTranspose2d( sin+2,sout, kernel_size, stride,pad,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

class ResNetEncoder(ModelResNet18) :
    def __init__(self, input_shape, latent_dim=32, pretrained=False, nbr_layer=4 ) :
        super(ResNetEncoder,self).__init__(input_shape=input_shape, 
                                           feature_dim=2*latent_dim, 
                                           nbr_layer=nbr_layer, 
                                           pretrained=pretrained)
        
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

    def save(self,path) :
        reswts = self.state_dict()
        respath = path + 'RESNETEncoder.weights'
        torch.save( reswts, respath )
        print('RESNETEncoder saved at : {}'.format(respath) )


    def load(self,path) :
        respath = path + 'RESNETEncoder.weights'
        self.load_state_dict( torch.load( respath ) )
        print('RESNETEncoder loaded from : {}'.format(respath) )


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
        #self.fc = coorddeconv( ind, outd, k, stride=1, pad=0, batchNorm=False)
        outd = ind 
        
        for i in reversed(range(self.net_depth)) :
            ind = outd
            outd = self.conv_dim*(2**i)
            self.dcs.append( coorddeconv( ind, outd,k,stride=stride,pad=pad) )
            self.dcs.append( nn.LeakyReLU(0.05) )
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
        self.dcout = coorddeconv( ind, outd, k, stride=stride, pad=pad, batchNorm=False)
        
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = z.repeat(1, 1, 4, 4)
        #out = F.leaky_relu( self.fc(z), 0.05)
        out = F.leaky_relu( self.dcs(out), 0.05)
        out = torch.sigmoid( self.dcout(out))
        return out

    def forward(self,z) :
        return self.decode(z)

    def save(self, path):
        dec_wts = self.state_dict()
        decpath = path + 'Decoder.weights'
        torch.save( dec_wts, decpath )
        print('Decoder saved at : {}'.format(decpath) )

    def load(self, path):
        decpath = path + 'Decoder.weights'
        self.decoder.load_state_dict( torch.load( decpath ) )
        print('Decoder loaded from : {}'.format(decpath) )


class ResNetBetaVAE(nn.Module) :
    def __init__(self, beta=1e4, 
                       latent_dim=32,
                       input_shape=[3, 64, 64], 
                       decoder_conv_dim=32, 
                       pretrained=False, 
                       resnet_nbr_layer=2,
                       decoder_nbr_layer=4,
                       NormalOutputDistribution=True,
                       EncodingCapacityStep=None,
                       maxEncodingCapacity=1000,
                       nbrEpochTillMaxEncodingCapacity=4,
                       constrainedEncoding=True,
                       observation_sigma=0.5):
        super(ResNetBetaVAE,self).__init__()

        self.beta = beta
        self.observation_sigma = observation_sigma
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.NormalOutputDistribution = NormalOutputDistribution

        self.EncodingCapacity = 0.0
        self.EncodingCapacityStep = EncodingCapacityStep
        self.maxEncodingCapacity = maxEncodingCapacity
        self.constrainedEncoding = constrainedEncoding
        self.nbrEpochTillMaxEncodingCapacity = nbrEpochTillMaxEncodingCapacity
        
        self.increaseEncodingCapacity = True
        if self.constrainedEncoding:
            nbritperepoch = 1e3
            print('ITER PER EPOCH : {}'.format(nbritperepoch))
            nbrepochtillmax = self.nbrEpochTillMaxEncodingCapacity
            nbrittillmax = nbrepochtillmax * nbritperepoch
            print('ITER TILL MAX ENCODING CAPACITY : {}'.format(nbrittillmax))
            self.EncodingCapacityStep = self.maxEncodingCapacity / nbrittillmax        

        self.encoder = ResNetEncoder(input_shape=input_shape, 
                                     latent_dim=latent_dim,
                                     nbr_layer=resnet_nbr_layer,
                                     pretrained=pretrained)
        self.decoder = Decoder(output_shape=input_shape,
                               latent_dim=latent_dim, 
                               net_depth=decoder_nbr_layer,
                               conv_dim=decoder_conv_dim)

    def get_feature_shape(self):
        return self.latent_dim
        
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

    def save(self,path) :
        self.encoder.save(path)
        self.decoder.save(path)

    def load(self,path) :
        self.encoder.load(path)
        self.decoder.load(path)

    def _forward(self,x=None,evaluation=False,fixed_latent=None,data=None) :
        if data is None and x is not None :
            if evaluation :
                self.z, self.mu, self.log_var = self.encodeZ(x) 
            else :
                self.VAE_output, self.mu, self.log_var = self(x)
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

        #--------------------------------------------------------------------------------------------------------------
        # VAE loss :
        #--------------------------------------------------------------------------------------------------------------
        # Reconstruction loss :
        if observation_sigma is not None:
            self.observation_sigma = observation_sigma
        if self.NormalOutputDistribution:
            #Normal :
            self.neg_log_lik = -Normal(self.VAE_output, self.observation_sigma).log_prob( gtx)
        else:
            #Bernoulli :
            self.neg_log_lik = -Bernoulli( self.VAE_output ).log_prob( gtx )
        
        self.reconst_loss = torch.sum( self.neg_log_lik.view( self.batch_size, -1), dim=1)
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        # KL Divergence :
        self.true_kl_divergence = 0.5 * (self.mu**2 + torch.exp(self.log_var) - self.log_var -1)
        
        if self.EncodingCapacityStep is None :
            self.kl_divergence = torch.sum(self.true_kl_divergence, dim=1)
            self.VAE_loss = self.reconst_loss + self.beta*self.kl_divergence
        else:
            self.kl_divergence_regularized =  torch.abs( torch.sum(self.true_kl_divergence, dim=1) - self.EncodingCapacity ) 
            self.kl_divergence =  torch.sum( self.true_kl_divergence, dim=1 )
            self.VAE_loss = self.reconst_loss + self.beta * self.kl_divergence_regularized
            
            if self.increaseEncodingCapacity and self.training:
                self.EncodingCapacity += self.EncodingCapacityStep
            if self.EncodingCapacity >= self.maxEncodingCapacity :
                self.increaseEncodingCapacity = False 
        #--------------------------------------------------------------------------------------------------------------
        return self.VAE_loss, self.neg_log_lik, self.kl_divergence_regularized, self.true_kl_divergence
