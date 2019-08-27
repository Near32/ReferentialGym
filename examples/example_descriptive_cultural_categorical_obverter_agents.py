#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import numpy as np 
import ReferentialGym

import torch
import torchvision
import torchvision.transforms as T 


def test_example_cultural_obverter_agents():
  seed = 40
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  # In[23]:


  rg_config = {
      "observability":            "partial", 
      "max_sentence_length":      20,
      "nbr_communication_round":  1,  
      "nbr_distractors":          0,
      "distractor_sampling":      "uniform",#"similarity-0.5",
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  

      "descriptive":              True,
      "descriptive_target_ratio": 0.5,
      # Default: 1-(1/(nbr_distractors+2)), 
      # otherwise the agent find the local minimum
      # where it only predicts 'no-target'...

      "object_centric":           False,
      
      "nbr_stimulus":             1,

      "graphtype":                'obverter', #'obverter'/reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      "tau0":                     0.2,
      "vocab_size":               5,

      "agent_architecture":       'CNN', #'CNN', #/'pretrained-ResNet18-2'

      "cultural_pressure_it_period": None,
      "cultural_substrate_size":  1,

      "iterated_learning_scheme": False,
      "iterated_learning_period": 200,
      
      "obverter_stop_threshold":  0.95,  #0.0 if not in use.
      "obverter_nbr_games_per_round": 20,

      "obverter_least_effort_loss": False,
      "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

      "batch_size":               256,
      "dataloader_num_worker":    2,
      "stimulus_depth_dim":       3,
      "stimulus_resize_dim":      64,#28,
      
      "learning_rate":            6e-4,
      "adam_eps":                 1e-5,

      "with_gradient_clip":       True,
      "gradient_clip":            1e-1,
      
      "with_utterance_penalization":  False,
      "utterance_penalization_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
                                                # The greater this value, the greater the loss/cost.
      "utterance_penalization_factor":    1e-2,

      "with_speaker_entropy_regularization":  False,
      "with_listener_entropy_regularization":  False,
      "with_weight_maxl1_loss":   False,

      "with_grad_logging":        True,
      "use_cuda":                 True,
  }

  assert( rg_config['observability'] == 'partial') # Descriptive scheme is always with partial observability...
  assert( rg_config['nbr_communication_round']==1) # In descriptive scheme, the multi-round/step communication scheme is not implemented yet.

  save_path = './'
  save_path += 'NLLLoss' #'MSELoss'
  save_path += '+UsingWIDX+GRU'
  #save_path += '+ProbOverDistrAndVocab-'
  if rg_config['with_utterance_penalization']:
    save_path += "+Tau-10-OOV{}Prob{}".format(rg_config['utterance_penalization_factor'], rg_config['utterance_penalization_oov_prob'])  
  if rg_config['with_gradient_clip']:
    save_path += '+ClipGrad{}'.format(rg_config['gradient_clip'])
  if rg_config['with_speaker_entropy_regularization']:
    save_path += 'SPEntrReg'
  if rg_config['with_listener_entropy_regularization']:
    save_path += 'LSEntrReg'
  if rg_config['iterated_learning_scheme']:
    save_path += '-ILM{}+ListEntrReg'.format(rg_config['iterated_learning_period'])
  save_path += '-{}Speaker-{}-{}{}CulturalCategoricalObverter{}-{}GPR-S{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}Distr{}-stim{}-vocab{}over{}_CIFAR10_{}'.format(rg_config['cultural_substrate_size'], 
    rg_config['cultural_pressure_it_period'],
    'ObjectCentric' if rg_config['object_centric'] else '',
    'Descriptive{}'.format(rg_config['descriptive_target_ratio']) if rg_config['descriptive'] else '',
    rg_config['obverter_stop_threshold'],
    rg_config['obverter_nbr_games_per_round'],
    seed,
    rg_config['observability'], 
    rg_config['batch_size'], 
    rg_config['learning_rate'],
    rg_config['graphtype'], 
    rg_config['tau0'], 
    rg_config['distractor_sampling'],
    rg_config['nbr_distractors'], 
    rg_config['nbr_stimulus'], 
    rg_config['vocab_size'], 
    rg_config['max_sentence_length'], 
    rg_config['agent_architecture'])

  rg_config['save_path'] = save_path

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agent Configuration:

  # In[3]:


  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors']
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['use_obverter_threshold_to_stop_message_generation'] = rg_config['obverter_stop_threshold']
  agent_config['descriptive'] = rg_config['descriptive']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  '''
  # CNN : from paper
  agent_config['cnn_encoder_channels'] = [32,32,32,32,32,32,32,32]
  agent_config['cnn_encoder_kernels'] = [3,3,3,3,3,3,3,3]
  agent_config['cnn_encoder_strides'] = [2,1,1,2,1,2,1,2]
  agent_config['cnn_encoder_paddings'] = [1,1,1,1,1,1,1,1]
  agent_config['cnn_encoder_feature_dim'] = 512
  agent_config['cnn_encoder_mini_batch_size'] = 128
  agent_config['temporal_encoder_nbr_hidden_units'] = 64
  agent_config['temporal_encoder_nbr_rnn_layers'] = 1
  agent_config['temporal_encoder_mini_batch_size'] = 128
  agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
  agent_config['symbol_processing_nbr_rnn_layers'] = 1
  '''
  if 'CNN' in agent_config['architecture']:
    # CNN : 
    agent_config['cnn_encoder_channels'] = [32,32,32,32]
    agent_config['cnn_encoder_kernels'] = [3,3,3,3]
    agent_config['cnn_encoder_strides'] = [1,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['cnn_encoder_feature_dim'] = 256
    agent_config['cnn_encoder_mini_batch_size'] = 128
    agent_config['temporal_encoder_nbr_hidden_units'] = 64
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 128
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResNet' in agent_config['architecture']:
    # ResNet18-2:
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    agent_config['cnn_encoder_feature_dim'] = 512
    agent_config['cnn_encoder_mini_batch_size'] = 128
    agent_config['temporal_encoder_nbr_hidden_units'] = 64
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 128
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  # # Basic Agents

  # ## Obverter Speaker:

  # In[4]:


  from ReferentialGym.agents import CategoricalObverterAgent


  # In[5]:


  batch_size = 4
  nbr_distractors = agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  bspeaker = CategoricalObverterAgent(kwargs=agent_config, 
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

  blistener = CategoricalObverterAgent(kwargs=agent_config, 
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
      "distractor_sampling":      rg_config['distractor_sampling'],
      "nbr_distractors":          rg_config['nbr_distractors'],
      "observability":            rg_config['observability'],
      "object_centric":           rg_config['object_centric'],
      "descriptive":              rg_config['descriptive'],
      "descriptive_target_ratio": rg_config['descriptive_target_ratio']
  }

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)

  # In[22]:

  nbr_epoch = 15
  refgame.train(prototype_speaker=bspeaker, 
                prototype_listener=blistener, 
                nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

if __name__ == '__main__':
    test_example_cultural_obverter_agents()
