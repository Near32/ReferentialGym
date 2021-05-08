import os 
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np 
from numpy import linalg as LA
from scipy.stats import spearmanr
from tqdm import tqdm 

import itertools
from concurrent.futures import ProcessPoolExecutor

eps = 1e-20

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    .. _Gumbel-Softmax Straight Through trick:
        https://arxiv.org/abs/1705.11192
    
    """
    
    if eps < 1e-10:
        warnings.warn("`eps` parameter is used to exclude inf to occur from the computation of -log(-log(u)) where u ~ U[0,1)+eps. Safer values are greater than 1e-10.")

    u = torch.rand_like(logits)*(0.999-eps)+eps
    gumbels = -torch.log( -torch.log(u))
    
    '''
    # Pytorch 1.4:
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    '''
    
    logits_gumbels = (logits + gumbels) / (tau+eps)  # ~Gumbel(logits,tau)
    y_soft = logits_gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    
    if torch.isnan(ret).any():
        import ipdb; ipdb.set_trace()

    return ret

class StraightThroughGumbelSoftmaxLayer(nn.Module):
  def __init__(self, 
               input_dim,
               inv_tau0=0.5):
    super(StraightThroughGumbelSoftmaxLayer, self).__init__()
    self.input_dim = input_dim
    self.inv_tau0 = inv_tau0
    self.tau_fc = nn.Sequential(
      nn.Linear(self.input_dim, 1,bias=False),
      nn.Softplus()
    )

  def forward(self,
              logits,
              param):
    batch_size = param.shape[0]
    tau = 1. / ( self.tau_fc(param.reshape(-1,self.input_dim))+self.inv_tau0).reshape(batch_size, -1)
    one_hot_distr = gumbel_softmax(
      logits=logits, 
      tau=tau, 
      hard=True, 
      dim=-1, 
      eps=1e-6)
    return one_hot_distr
                    



def cardinality(data):
    if isinstance(data[0], np.ndarray):
        data_array = np.concatenate([np.expand_dims(d, 0) for d in data], axis=0)
        data_set = np.unique(data_array, axis=0)
    else:
        data_set = set(data)
    return len(data_set)


# https://www.python-course.eu/levenshtein_distance.php
def compute_levenshtein_distance(s1, s2):
    rows = len(s1)+1
    cols = len(s2)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i
    
    # From there, we can compute iteratively how many steps
    # are needed to transform the source prefix (at col) into
    # the target prefix (at row):
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            dist[i][j] = min(dist[i-1][j] + 1,      # deletion
                                 dist[i][j-1] + 1,      # insertion
                                 dist[i-1][j-1] + cost) # substitution
    return float(dist[-1][-1])


def compute_cosine_sim(v1, v2):
    global eps
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    v1_norm = LA.norm(v1)+eps
    v2_norm = LA.norm(v2)+eps
    cos_sim = np.matmul(v1/v1_norm,(v2/v2_norm).transpose())
    return cos_sim

def compute_levenshtein_distance_for_idx_over_comprange(sentences, idx, comprange=None):
    if comprange is None: 
      comprange = len(sentences)
    else:
      # ratio:
      comprange = int(comprange*len(sentences))
    levs = []
    s1 = sentences[idx]
    tillidx = min(len(sentences)-1,idx+1+comprange)
    for idx2, s2 in enumerate(sentences[idx+1:tillidx]): 
        levs.append( compute_levenshtein_distance(s1,s2))
    return levs 

def compute_cosine_sim_for_idx_over_comprange(features, idx, comprange=None):
    if comprange is None: 
      comprange = len(features)
    else:
      # ratio:
      comprange = int(comprange*len(features))
    cossims = []
    f1 = features[idx]
    tillidx = min(len(features)-1,idx+1+comprange)
    for idx2, f2 in enumerate(features[idx+1:tillidx]): 
        cossims.append( compute_cosine_sim(f1.squeeze(),f2.squeeze()))
    return cossims

def compute_topographic_similarity_parallel(sentences,features,comprange=None, max_workers=32):
    executor = ProcessPoolExecutor(max_workers=max_workers)
    indices = list(range(len(sentences)))
    levs = []
    #for idx1, idx1_levs in tqdm(zip(indices, executor.map(compute_levenshtein_distance_for_idx_over_comprange, itertools.repeat(sentences), indices, itertools.repeat(comprange)))):
    for idx1, idx1_levs in zip(indices, executor.map(compute_levenshtein_distance_for_idx_over_comprange, itertools.repeat(sentences), indices, itertools.repeat(comprange))):
        for l in idx1_levs: 
            levs.append(l)

    indices = list(range(len(features)))
    cossims = []
    #for idx1, idx1_cossims in tqdm(zip(indices, executor.map(compute_cosine_sim_for_idx_over_comprange, itertools.repeat(features), indices, itertools.repeat(comprange)))):
    for idx1, idx1_cossims in zip(indices, executor.map(compute_cosine_sim_for_idx_over_comprange, itertools.repeat(features), indices, itertools.repeat(comprange))):
        for c in idx1_cossims: 
            cossims.append(c)
    
    rho, p = spearmanr(levs, cossims)
    return -rho, p, levs, cossims

def query_vae_latent_space(omodel, sample, path, minibatch_size=2, test=False, full=True, idxoffset=None, suffix='', use_cuda=False):
  if use_cuda:
    model = omodel.cuda()
  else:
    model = omodel.cpu()

  training = model.training
  model.train(False)
  
  z_dim = model.get_feature_shape()
  img_depth=model.input_shape[0]
  img_dim = model.input_shape[1]
  
  fixed_x = sample.view(-1, img_depth, img_dim, img_dim)
  if use_cuda:  
    fixed_x = fixed_x.cuda()
  else:
    fixed_x = fixed_x.cpu()

  # variations over the latent variable :
  sigma_mean = 3.0*torch.ones((z_dim)) #args.queryVAR*torch.ones((z_dim))
  mus = []
  for stin in torch.split(fixed_x, split_size_or_sections=minibatch_size, dim=0):
      _, mu, _ = model.encodeZ(stin)
      mus.append(mu.cpu())
  mus = torch.cat(mus, dim=0)
  #z, mu, log_sig_sq  = model.encodeZ(fixed_x)
  mu_mean = mus.cpu().data[0]#.unsqueeze(0)
  #print(z,mu_mean,sigma_mean)
  
  # Save generated variable images :
  nbr_steps = 8 #args.querySTEPS
  gen_images = list()

  if (z_dim <= 50) or full:
    for latent in range(z_dim):
      var_z0 = torch.zeros((nbr_steps, z_dim), requires_grad=False)
      val = mu_mean[latent]-sigma_mean[latent]
      step = 2.0*sigma_mean[latent]/nbr_steps
      for i in range(nbr_steps) :
        var_z0[i] = mu_mean
        var_z0[i][latent] = val
        val += step

      if use_cuda: var_z0 = var_z0.cuda()

      gen_images_latent = []
      for stin in torch.split(var_z0, split_size_or_sections=minibatch_size, dim=0):
          gil = model.decode(stin)
          gen_images_latent.append(gil.cpu())
      gen_images_latent = torch.cat(gen_images_latent, dim=0)
      #gen_images_latent = model.decode(var_z0)
      npfx = gen_images_latent.cpu().data 
      
      gen_images.append(gen_images_latent)
    
    gen_images = torch.cat(gen_images, dim=0)
    grid_gen_images = torchvision.utils.make_grid(gen_images, nrow=nbr_steps)

    save_path = os.path.join(path, 'generated_images/')
    if test :
      save_path = os.path.join(save_path, 'test/')
    os.makedirs(save_path, exist_ok=True)
    save_path += 'query{}{}.png'.format(idxoffset, suffix)
    torchvision.utils.save_image(gen_images, save_path )
  
  reconst_images = []
  for stin in torch.split(fixed_x, split_size_or_sections=minibatch_size, dim=0):
      model_outputs = model(stin)
      reconst_images.append(model_outputs[0].cpu())
  reconst_images = torch.cat(reconst_images, dim=0)
  
  npfx = reconst_images.cpu().data
  orimg = fixed_x.cpu().data
  ri = torch.cat( [orimg, npfx], dim=2)
  save_path = os.path.join(path, 'reconstructed_images/')
  if test :
    save_path = os.path.join(save_path, 'test/')
  os.makedirs(save_path, exist_ok=True)
  query_save_path = save_path + 'query{}{}.png'.format(idxoffset, suffix)
  torchvision.utils.save_image(ri,query_save_path )

  if hasattr(model.encoder, 'attention_weights'):
    attention_weights = model.encoder.attention_weights
    seq_len = len(attention_weights)
    if isinstance(attention_weights, list):
      attention = torch.cat([ attention[0].unsqueeze(0) for attention in attention_weights], dim=0).cpu().data.transpose(1,3).unsqueeze(2)
    else:
      attention = attention_weights
    # seq_len x nbr_slots x 1 x spatialDim x spatialDim 
    spatialDim = attention.size(-1)
    # rescale between [0.25, 1.0]:
    attention = 0.75*attention + 0.25
    nbr_slot = attention.size(1)
    if seq_len != orimg.size(0):
      orimg = fixed_x[0].cpu().data.unsqueeze(0).repeat(seq_len*nbr_slot, 1, 1, 1)
    else:
      orimg = fixed_x.cpu().data.unsqueeze(0).repeat(nbr_slot, 1, 1, 1, 1)
      orimg = orimg.transpose(1,0).contiguous().view((-1, *fixed_x.size()[1:]))
    # seq_len x 3 x img_w x img_h 

    # resize: if needs be:
    imgw = orimg.size(-1)
    if imgw != spatialDim:
      resize = torchvision.transforms.Compose( [torchvision.transforms.ToPILImage(), 
        torchvision.transforms.Resize(imgw),
        torchvision.transforms.ToTensor()])
      attention = attention.contiguous().view(seq_len*nbr_slot, 1, spatialDim, spatialDim)
      attention = torch.cat([ resize(attention[i]).unsqueeze(0) for i in range(seq_len*nbr_slot)], dim=0)
    attention = attention.contiguous().view(seq_len*nbr_slot, 1, imgw, imgw)
    # seq_len*nbr_slot x 1 x imgw x imgw 
    attention = attention.contiguous().repeat(1,3,1,1).cpu().data

    att_img = attention * orimg
    grid_att_img = torchvision.utils.make_grid(att_img, nrow=nbr_slot)
    att_save_path = save_path+'att{}{}.png'.format(idxoffset, suffix)
    torchvision.utils.save_image(grid_att_img, att_save_path)

  if hasattr(model.encoder, 'attention_reconstructions'):
    attention_reconstructions = model.encoder.attention_reconstructions
    seq_len = len(attention_weights)
    if isinstance(attention_weights, list):
      recs = torch.cat([ attention[0].unsqueeze(0) for attention in attention_reconstructions], dim=0).cpu().data.transpose(1,3).unsqueeze(2)
    else:
      recs = attention_reconstructions
    # seq_len x nbr_slots x 3 x spatialDim x spatialDim 
    spatialDim = recs.size(-1)
    nbr_slot = recs.size(1)
    
    imgw = orimg.size(-1)
    if imgw != spatialDim:
      resize = torchvision.transforms.Compose( [torchvision.transforms.ToPILImage(), 
        torchvision.transforms.Resize(imgw),
        torchvision.transforms.ToTensor()])
      recs = recs.contiguous().view(seq_len*nbr_slot, 1, spatialDim, spatialDim)
      recs = torch.cat([ resize(recs[i]).unsqueeze(0) for i in range(seq_len*nbr_slot)], dim=0)
    recs = recs.contiguous().view(seq_len*nbr_slot, -1, imgw, imgw)
    # seq_len*nbr_slot x 3 x imgw x imgw 
    recs = recs.contiguous()

    grid_recs = torchvision.utils.make_grid(recs, nrow=nbr_slot)
    recs_save_path = save_path+'recs{}{}.png'.format(idxoffset, suffix)
    torchvision.utils.save_image(grid_recs, recs_save_path)

    model.train(training)


"""
Based on:
https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L65
"""

class PositionalEncoding(nn.Module):
  r"""Inject some information about the relative or absolute position of the tokens
      in the sequence. The positional encodings have the same dimension as
      the embeddings, so that the two can be summed. Here, we use sine and cosine
      functions of different frequencies.
  .. math::
      \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
      \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
      \text{where pos is the word position and i is the embed idx)
  Args:
      d_model: the embed dim (required).
      dropout: the dropout value (default=0.1).
      max_len: the max. length of the incoming sequence (default=5000).
  Examples:
      >>> pos_encoder = PositionalEncoding(d_model)
  """

  def __init__(
    self, 
    d_model, 
    dropout=0.1, 
    max_len=5000):
    
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    # (max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # (d_model//2, )
    pe[:, 0::2] = torch.sin(position * div_term)
    # alternating from first item...
    pe[:, 1::2] = torch.cos(position * div_term)
    # alternating from second item...
    pe = pe.unsqueeze(0)#.transpose(0, 1)
    # (1, max_len, d_model)
    self.register_buffer('pe', pe)

  def forward(self, x):
      r"""Inputs of forward function
      Args:
          x: the sequence fed to the positional encoder model (required).
      Shape:
          x: [ batch size, sequence length, embed dim]
          output: [batch size, sequence length, embed dim]
      Examples:
          >>> output = pos_encoder(x)
      """

      x = x + self.pe[:, :x.size(1), :].to(x.device)
      # (batch_size, sentence_length, d_model)
      return self.dropout(x)