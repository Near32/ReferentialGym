import os 
import torch
import torchvision
import numpy as np 
from numpy import linalg as LA
from scipy.stats import spearmanr
from tqdm import tqdm 


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

    u = torch.rand_like(logits)+eps
    gumbels = -torch.log( -u.log())
    #gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


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
    v1_norm = LA.norm(v1)
    v2_norm = LA.norm(v2)
    cos_sim = np.matmul(v1/v1_norm,(v2/v2_norm).transpose())
    return cos_sim


def compute_topographic_similarity(sentences,features,comprange=100):
    levs = []
    for idx1 in tqdm(range(len(sentences))):
        s1 = sentences[idx1]
        tillidx = min(len(sentences)-1,idx1+1+comprange)
        for idx2, s2 in enumerate(sentences[idx1+1:tillidx]): 
            levs.append( compute_levenshtein_distance(s1,s2))
    cossims = []
    for idx1 in tqdm(range(len(features))):
        f1 = features[idx1]
        tillidx = min(len(sentences)-1,idx1+1+comprange)
        for idx2, f2 in enumerate(features[idx1+1:tillidx]): 
            cossims.append( compute_cosine_sim(f1,f2))
    
    rho, p = spearmanr(levs, cossims)
    return -rho, p, levs, cossims  


def query_vae_latent_space(omodel, sample, path, test=False, full=True, idxoffset=None, suffix='', use_cuda=False):
  if use_cuda:
    model = omodel.cuda()
  else:
    model = omodel.cpu()

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
  z, mu, log_sig_sq  = model.encodeZ(fixed_x)
  mu_mean = mu.cpu().data[0]#.unsqueeze(0)
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

      gen_images_latent = model.decoder(var_z0)
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
    
  reconst_images,_ ,_ = model(fixed_x)
  
  npfx = reconst_images.cpu().data
  orimg = fixed_x
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
    attention = torch.cat([ attention[0].unsqueeze(0) for attention in attention_weights], dim=0).cpu().data.transpose(1,3).unsqueeze(2)
    # seq_len x nbr_slots x 1 x spatialDim x spatialDim 
    spatialDim = attention.size(-1)
    # rescale between [0.25, 1.0]:
    attention = 0.75*attention + 0.25
    nbr_slot = attention.size(1)
    orimg = orimg[0].unsqueeze(0).repeat(seq_len*nbr_slot, 1, 1, 1).cpu().data
    # seq_len x 3 x img_w x img_h 

    # resize:
    imgw = orimg.size(-1)
    resize = torchvision.transforms.Compose( [torchvision.transforms.ToPILImage(), 
      torchvision.transforms.Resize(imgw),
      torchvision.transforms.ToTensor()])
    rattention = attention.contiguous().view(seq_len*nbr_slot, 1, spatialDim, spatialDim)
    rattention = torch.cat([ resize(rattention[i]).unsqueeze(0) for i in range(seq_len*nbr_slot)], dim=0)
    rattention = rattention.contiguous().view(seq_len*nbr_slot, 1, imgw, imgw)
    # seq_len*nbr_slot x 1 x imgw x imgw 
    rattention = rattention.contiguous().repeat(1,3,1,1)

    att_img = rattention * orimg
    grid_att_img = torchvision.utils.make_grid(att_img, nrow=nbr_slot)
    att_save_path = save_path+'att{}{}.png'.format(idxoffset, suffix)
    torchvision.utils.save_image(grid_att_img, att_save_path)


def permutate_latents(z):
    assert(z.dim() == 2)
    batch_size, latent_dim = z.size()
    pz = list()
    for lz in torch.split(z, split_size_or_sections=1, dim=1):
        b_perm = torch.randperm(batch_size).to(z.device)
        p_lz = lz[b_perm]
        pz.append(p_lz)
    pz = torch.cat(pz, dim=1)
    return pz 