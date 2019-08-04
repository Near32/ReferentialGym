#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ReferentialGym

import torch
import torchvision
import torchvision.transforms as T 

def test_example_basic_agents():
  # # Hyperparameters:

  # In[23]:


  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      10,
      "nbr_communication_round":  1,
      "nbr_distractors":          31,
      "distractor_sampling":      "uniform",
      "descriptive":              False,
      "object_centric":           False,
      "nbr_stimulus":             1,

      "graphtype":                'straight_through_gumbel_softmax', #'reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      "tau0":                      0.2,
      "vocab_size":               100,

      "cultural_pressure_it_period": 1000,
      "cultural_substrate_size":  5,
      
      "batch_size":               64,
      "dataloader_num_worker":    8,
      "stimulus_resize_dim":      64,#28,
      
      "learning_rate":            3e-4,
      "adam_eps":                 1e-5,
      "gradient_clip":            5,
      "with_weight_maxl1_loss":   False,

      "use_cuda":                 True,
  }


  # # Agent Configuration:

  # In[3]:


  speaker_config = dict()
  speaker_config['nbr_distractors'] = 0 if rg_config['observability'] == "partial" else rg_config['nbr_distractors']
  speaker_config['nbr_stimulus'] = rg_config['nbr_stimulus']

  # Recurrent Convolutional Architecture:
  #speaker_config['architecture'] = 'CNN'
  speaker_config['architecture'] = 'ResNet18-2'
  speaker_config['cnn_encoder_channels'] = [32, 32, 64]
  speaker_config['cnn_encoder_kernels'] = [4, 3, 3]
  speaker_config['cnn_encoder_strides'] = [4, 2, 1]
  speaker_config['cnn_encoder_paddings'] = [0, 1, 1]
  speaker_config['cnn_encoder_feature_dim'] = 512
  speaker_config['cnn_encoder_mini_batch_size'] = 128
  speaker_config['temporal_encoder_nbr_hidden_units'] = 512
  speaker_config['temporal_encoder_nbr_rnn_layers'] = 1
  speaker_config['temporal_encoder_mini_batch_size'] = 128
  speaker_config['symbol_processing_nbr_hidden_units'] = 512
  speaker_config['symbol_processing_nbr_rnn_layers'] = 1

  import copy
  listener_config = copy.deepcopy(speaker_config)
  listener_config['nbr_distractors'] = rg_config['nbr_distractors']


  # # Basic Agents

  # ## Basic Speaker:

  # In[4]:


  from ReferentialGym.agents import BasicCNNSpeaker


  # In[5]:


  batch_size = 4
  nbr_distractors = speaker_config['nbr_distractors']
  nbr_stimulus = speaker_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus,1,rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  bspeaker = BasicCNNSpeaker(kwargs=speaker_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length)

  print("Speaker:",bspeaker)

  # ## Basic Listener:

  # In[7]:


  from ReferentialGym.agents import BasicCNNListener


  # In[8]:


  batch_size = 4
  nbr_distractors = listener_config['nbr_distractors']
  nbr_stimulus = listener_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus,1,rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  blistener = BasicCNNListener(kwargs=listener_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length)

  print("Listener:",blistener)

  # # MNIST Dataset:

  # In[10]:

  from ReferentialGym.networks.utils import ResizeNormalize
  transform = ResizeNormalize(size=rg_config['stimulus_resize_dim'], normalize_rgb_values=False)
  #transform = T.ToTensor()

  #dataset = torchvision.datasets.MNIST(root='./datasets/', train=True, transform=None, target_transform=None, download=True)
  dataset = torchvision.datasets.MNIST(root='./datasets/', train=True, transform=transform, target_transform=None, download=False)
  dataset_args = {
      "dataset_class":            "LabeledDataset",
      "dataset":                  dataset,
      "nbr_stimulus":             rg_config['nbr_stimulus'],
      "nbr_distractors":          rg_config['nbr_distractors'],
  }

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)


  # In[20]:


  from tensorboardX import SummaryWriter
  logger = SummaryWriter('./{}_example_log'.format(speaker_config['architecture']))


  # In[22]:

  nbr_epoch = 100
  refgame.train(prototype_speaker=bspeaker, 
                prototype_listener=blistener, 
                nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

if __name__ == '__main__':
    test_example_basic_agents()