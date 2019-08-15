#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ReferentialGym

import torch
import torchvision
import torchvision.transforms as T 

def test_example_cultural_obverter_agents():
  seed = 50
  torch.manual_seed(seed)
  # # Hyperparameters:

  # In[23]:


  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      5,
      "nbr_communication_round":  1,
      "nbr_distractors":          31,
      "distractor_sampling":      "uniform",
      
      "descriptive":              False,
      "descriptive_target_ratio": 0.0,
      
      "object_centric":           False,
      
      "nbr_stimulus":             1,

      #"graphtype":                'obverter', #'obverter'/reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      "graphtype":                'straight_through_gumbel_softmax_obverter', #'obverter'/reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      
      "tau0":                     0.2,
      "vocab_size":               25,

      "agent_architecture":       'ResNet18-2', #'CNN'/'ResNet18-2'

      "cultural_pressure_it_period": None,
      "cultural_substrate_size":  1,
      
      "batch_size":               128,
      "dataloader_num_worker":    8,
      "stimulus_depth_dim":       3,
      "stimulus_resize_dim":      64,#28,
      
      "learning_rate":            3e-4,
      "adam_eps":                 1e-5,
      "gradient_clip":            50,
      "with_weight_maxl1_loss":   False,

      "use_cuda":                 True,
  }

  save_path = './{}Speaker-{}-STGTCulturalObverter-S{}-CELoss+SiSentEnc_{}_b{}-obs-{}-tau0-{}-distr{}-stim{}-vocab{}over{}_CIFAR10_{}_example_log'.format(rg_config['cultural_substrate_size'], 
    rg_config['cultural_pressure_it_period'],
    seed,rg_config['observability'], 
    rg_config['batch_size'], 
    rg_config['graphtype'], 
    rg_config['tau0'], 
    rg_config['nbr_distractors'], 
    rg_config['nbr_stimulus'], 
    rg_config['vocab_size'], 
    rg_config['max_sentence_length'], 
    rg_config['agent_architecture'])

  rg_config['save_path'] = save_path

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=1000)
  
  # # Agent Configuration:

  # In[3]:


  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors']
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['use_obverter_threshold_to_stop_message_generation'] = False

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['cnn_encoder_channels'] = [32, 32, 64]
  agent_config['cnn_encoder_kernels'] = [4, 3, 3]
  agent_config['cnn_encoder_strides'] = [4, 2, 1]
  agent_config['cnn_encoder_paddings'] = [0, 1, 1]
  agent_config['cnn_encoder_feature_dim'] = 512
  agent_config['cnn_encoder_mini_batch_size'] = 32
  agent_config['temporal_encoder_nbr_hidden_units'] = 512
  agent_config['temporal_encoder_nbr_rnn_layers'] = 1
  agent_config['temporal_encoder_mini_batch_size'] = 32
  agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
  agent_config['symbol_processing_nbr_rnn_layers'] = 1

  # # Basic Agents

  # ## Obverter Speaker:

  # In[4]:


  from ReferentialGym.agents import ObverterAgent


  # In[5]:


  batch_size = 4
  nbr_distractors = agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  bspeaker = ObverterAgent(kwargs=agent_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='os0',
                                logger=logger)

  print("Speaker:",bspeaker)

  # ## Obverter Listener:

  # In[7]:

  batch_size = 4
  nbr_distractors = agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  blistener = ObverterAgent(kwargs=agent_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='ol0',
                                logger=logger)

  print("Listener:",blistener)

  # # Dataset:

  # In[10]:

  from ReferentialGym.networks.utils import ResizeNormalize
  transform = ResizeNormalize(size=rg_config['stimulus_resize_dim'], normalize_rgb_values=False)
  #transform = T.ToTensor()

  #dataset = torchvision.datasets.MNIST(root='./datasets/MNIST/', train=True, transform=transform, target_transform=None, download=True)
  train_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/', train=True, transform=transform, target_transform=None, download=True)
  test_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/', train=False, transform=transform, target_transform=None, download=True)
  dataset_args = {
      "dataset_class":            "LabeledDataset",
      "train_dataset":            train_dataset,
      "test_dataset":             test_dataset,
      "nbr_stimulus":             rg_config['nbr_stimulus'],
      "nbr_distractors":          rg_config['nbr_distractors'],
      "observability":            rg_config['observability'],
      "descriptive":              rg_config['descriptive'],
      "descriptive_target_ratio": rg_config['descriptive_target_ratio']
  }

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)

  # In[22]:

  nbr_epoch = 100
  refgame.train(prototype_speaker=bspeaker, 
                prototype_listener=blistener, 
                nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

if __name__ == '__main__':
    test_example_cultural_obverter_agents()
