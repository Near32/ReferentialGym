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
  torch.autograd.set_detect_anomaly(True)
  seed = 20
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  # In[23]:


  rg_config = {
      "observability":            "partial", 
      "max_sentence_length":      5,
      "nbr_communication_round":  1,  
      "nbr_distractors":          7,
      "distractor_sampling":      "uniform",#"similarity-0.98",#"similarity-0.75",
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  
      
      "descriptive":              True,
      "descriptive_target_ratio": 0.97, 
      # Default: 1-(1/(nbr_distractors+2)), 
      # otherwise the agent find the local minimum
      # where it only predicts 'no-target'...

      "object_centric":           False,
      
      "nbr_stimulus":             1,

      "graphtype":                'obverter', #'[informed-]obverter'/reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      "tau0":                     0.1,
      "vocab_size":               10,

      "agent_architecture":       'BetaVAE', #'CNN', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'

      "cultural_pressure_it_period": None,
      "cultural_speaker_substrate_size":  1,
      "cultural_listener_substrate_size":  1,
      "cultural_reset_strategy":  "oldestL", # "uniformSL" #"meta-oldestL-SGD"
      "cultural_reset_meta_learning_rate":  1e-3,

      "iterated_learning_scheme": False,
      "iterated_learning_period": 200,

      "obverter_stop_threshold":  0.95,  #0.0 if not in use.
      "obverter_nbr_games_per_round": 2,

      "obverter_least_effort_loss": False,
      "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

      "batch_size":               128,
      "dataloader_num_worker":    8,
      "stimulus_depth_dim":       1, #3,
      "stimulus_resize_dim":      32,#28,
      
      "learning_rate":            6e-4,
      "adam_eps":                 1e-8, #1e-5
      "dropout_prob":             0.0,
      
      "use_homoscedastic_multitasks_loss": False,

      "use_curriculum_nbr_distractors": False,
      "curriculum_distractors_window_size": 25, #100,

      "with_gradient_clip":       False,
      "gradient_clip":            1e-1,

      "unsupervised_segmentation_factor": None, #1e5
      "nbr_experience_repetition":  1,

      "with_utterance_penalization":  False,
      "with_utterance_promotion":     False,
      "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
                                                # The greater this value, the greater the loss/cost.
      "utterance_factor":    1e-2,

      "with_speaker_entropy_regularization":  False,
      "with_listener_entropy_regularization":  False,
      "entropy_regularization_factor":    2e0,

      "with_mdl_principle":       False,
      "mdl_principle_factor":     5e-2,

      "with_weight_maxl1_loss":   False,

      "with_grad_logging":        False,
      "use_cuda":                 True,
  }

  assert( rg_config['observability'] == 'partial') # Descriptive scheme is always with partial observability...
  assert( rg_config['nbr_communication_round']==1) # In descriptive scheme, the multi-round/step communication scheme is not implemented yet.

  assert( abs(rg_config['descriptive_target_ratio']-(1-1.0/(rg_config['nbr_distractors']+2))) <= 1e-1)

  #vae_beta = 5e-1
  #vae_beta = 1e2
  
  # Factor VAE:
  gaussian = False 
  vae_observation_sigma = 0.25 #0.11
  
  vae_beta = 1e0
  factor_vae_gamma = 0.0 #10 #6.4

  monet_gamma = 5e-1
  
  vae_constrainedEncoding = False
  maxCap = 1e3 #1e2
  nbrepochtillmaxcap = 2
  skip_interval = 48
  
  #save_path = './FashionMNIST+LVAE+RDec'
  
  #save_path = './SoC+L6VAE+BrDec+AttPrior'
  
  #save_path = './SoC-CNN+Det'#tf-mlp-fbg-decNS-tnormal-SoC+DualVAEL+tiny'
  #save_path = './MRL20-CNN-mhdpa+Det'
  
  #save_path = './MRL20-SAttCNN+CNN+D'
  #save_path = './FVAE/FashionMNIST-CNN+D+FactorVAE'

  dsprites_divider = 200
  dsprites_offset = 2
  #save_path = f'./FVAE/dSprites-ttsplit{dsprites_divider}-{dsprites_offset}-Det+VAE+BernBCE+repA'
  #save_path = f'./FVAE/dSprites-ttsplit{dsprites_divider}-{dsprites_offset}-VAE+BernBCE+repA'
  #save_path = f'./FVAE/dSprites-ttsplit{dsprites_divider}-{dsprites_offset}-LargeCNN+BernBCE+repA'
  
  #save_path = f'./test_TCD_Topo/dSprites-ttsplit{dsprites_divider}-{dsprites_offset}-LargeCNN+repA'
  save_path = f'./test_TCD_Topo/dSprites-ttsplit{dsprites_divider}-{dsprites_offset}-BetaVAE_CNN+repA'
  
  #save_path = './FVAE/dSprites-CNN+D+FactorVAE+RS+MSE'
  #save_path = './FVAE/dSprites-CNN+D+FactorVAE+MEANMSE'
  
  #save_path = './MineRL-S{}+BetaVAE+BrDec'.format(skip_interval)
  
  #save_path += 'TF64-NoSW+VSS-SDP{}'.format(rg_config['dropout_prob'])
  #save_path += 'NLLLoss' #'MSELoss'
  #save_path += '+UsingWIDX+GRU+Logit4DistrTarNoTarg'
  #save_path += 'CPtau05e0+1e1LeastEffort+5e1'
  
  if rg_config['use_curriculum_nbr_distractors']:
    #save_path += '+Curr'
    save_path += f"+W{rg_config['curriculum_distractors_window_size']}Curr"
  if rg_config['use_homoscedastic_multitasks_loss']:
    save_path += '+Homo'
  if rg_config['with_utterance_penalization']:
    save_path += "+Tau-10-OOV{}PenProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
  if rg_config['with_utterance_promotion']:
    save_path += "+Tau-10-OOV{}ProProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
  
  if rg_config['with_gradient_clip']:
    save_path += '+ClipGrad{}'.format(rg_config['gradient_clip'])
  
  if rg_config['with_speaker_entropy_regularization']:
    save_path += 'SPEntrReg{}'.format(rg_config['entropy_regularization_factor'])
  if rg_config['with_listener_entropy_regularization']:
    save_path += 'LSEntrReg{}'.format(rg_config['entropy_regularization_factor'])
  
  if rg_config['iterated_learning_scheme']:
    save_path += '-ILM{}+ListEntrReg'.format(rg_config['iterated_learning_period'])
  
  if rg_config['with_mdl_principle']:
    save_path += '-MDL{}'.format(rg_config['mdl_principle_factor'])
  
  if rg_config['cultural_pressure_it_period'] != 'None':  
    save_path += '-S{}L{}-{}-Reset{}'.\
      format(rg_config['cultural_speaker_substrate_size'], 
      rg_config['cultural_listener_substrate_size'],
      rg_config['cultural_pressure_it_period'],
      rg_config['cultural_reset_strategy']+str(rg_config['cultural_reset_meta_learning_rate']) if 'meta' in rg_config['cultural_reset_strategy'] else rg_config['cultural_reset_strategy'])
  
  save_path += '-{}{}CulturalDiffObverter{}-{}GPR-S{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}Distr{}-stim{}-vocab{}over{}_{}{}'.\
    format(
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
    rg_config['agent_architecture'],
    f"beta{vae_beta}-factor{factor_vae_gamma}" if 'BetaVAE' in rg_config['agent_architecture'] else '')

  save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in rg_config['agent_architecture'] else ''
  save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
  save_path += f"UnsupSeg{rg_config['unsupervised_segmentation_factor']}Rep{rg_config['nbr_experience_repetition']}" if rg_config['unsupervised_segmentation_factor'] is not None else ''
  
  rg_config['save_path'] = save_path

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agent Configuration:

  # In[3]:


  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['homoscedastic_multitasks_loss'] = rg_config['use_homoscedastic_multitasks_loss']
  agent_config['max_sentence_length'] = rg_config['max_sentence_length']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors']
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['use_obverter_threshold_to_stop_message_generation'] = True
  agent_config['descriptive'] = rg_config['descriptive']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['dropout_prob'] = rg_config['dropout_prob']
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
  '''
  if 'CNN' == agent_config['architecture']:
    # CNN : 
    agent_config['cnn_encoder_channels'] = [32,32,32] #[32,32,32,32]
    agent_config['cnn_encoder_kernels'] = [4,3,3]
    agent_config['cnn_encoder_strides'] = [2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1]
    agent_config['cnn_encoder_feature_dim'] = 256
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  '''
  if 'CNN' == agent_config['architecture']:
    # CNN : 
    agent_config['cnn_encoder_channels'] = [32,32,64] #[32,32,32,32]
    agent_config['cnn_encoder_kernels'] = [8,4,3]
    agent_config['cnn_encoder_strides'] = [2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1]
    agent_config['cnn_encoder_feature_dim'] = 256
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'BetaVAE' in agent_config['architecture']:
    agent_config['vae_nbr_latent_dim'] = 32
    agent_config['vae_decoder_nbr_layer'] = 3#4
    agent_config['vae_decoder_conv_dim'] = 32
    
    agent_config['cnn_encoder_feature_dim'] = agent_config['vae_nbr_latent_dim']
    
    agent_config['vae_beta'] = vae_beta
    agent_config['factor_vae_gamma'] = factor_vae_gamma
    agent_config['vae_use_gaussian_observation_model'] = gaussian 
    agent_config['vae_constrainedEncoding'] = vae_constrainedEncoding 
    agent_config['vae_max_capacity'] = maxCap
    agent_config['vae_nbr_epoch_till_max_capacity'] = nbrepochtillmaxcap
    agent_config['vae_tc_discriminator_hidden_units'] = tuple([2*agent_config['cnn_encoder_feature_dim']]*4+[2])
    
    agent_config['cnn_encoder_mini_batch_size'] = rg_config['batch_size']
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'MONet' in agent_config['architecture']:
    agent_config['unsup_seg_factor'] = rg_config['unsupervised_segmentation_factor']

    agent_config['monet_gamma'] = monet_gamma
    agent_config['monet_nbr_attention_slot'] = 8#10
    agent_config['monet_anet_block_depth'] = 2 #3

    agent_config['vae_nbr_latent_dim'] = 10
    agent_config['vae_decoder_nbr_layer'] = 3 #4
    agent_config['vae_decoder_conv_dim'] = 32
    agent_config['vae_observation_sigma'] = vae_observation_sigma
    
    agent_config['cnn_encoder_feature_dim'] = agent_config['vae_nbr_latent_dim']*agent_config['monet_nbr_attention_slot']
    
    agent_config['vae_beta'] = vae_beta
    agent_config['factor_vae_gamma'] = factor_vae_gamma
    agent_config['vae_constrainedEncoding'] = vae_constrainedEncoding 
    agent_config['vae_max_capacity'] = maxCap
    agent_config['vae_nbr_epoch_till_max_capacity'] = nbrepochtillmaxcap
    agent_config['vae_tc_discriminator_hidden_units'] = tuple([2*agent_config['cnn_encoder_feature_dim']]*4+[2])
    
    agent_config['cnn_encoder_mini_batch_size'] = rg_config['batch_size']
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResidualPABetaVAE' in agent_config['architecture'] or 'PHDPABetaVAE' in agent_config['architecture']:
    # ResNet18-2:
    '''
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    '''
    agent_config['vae_nbr_attention_slot'] = 10
    agent_config['vae_nbr_latent_dim'] = 6
    agent_config['vae_decoder_nbr_layer'] = 3
    agent_config['vae_decoder_conv_dim'] = 32
    
    agent_config['cnn_encoder_feature_dim'] = agent_config['vae_nbr_latent_dim']*agent_config['vae_nbr_attention_slot']
    
    agent_config['vae_beta'] = vae_beta
    agent_config['factor_vae_gamma'] = factor_vae_gamma
    agent_config['vae_max_capacity'] = maxCap
    agent_config['vae_nbr_epoch_till_max_capacity'] = nbrepochtillmaxcap
    agent_config['vae_tc_discriminator_hidden_units'] = tuple([2*agent_config['cnn_encoder_feature_dim']]*4+[2])
    
    agent_config['cnn_encoder_mini_batch_size'] = rg_config['batch_size']
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResNet' in agent_config['architecture'] and not('MHDPA' in agent_config['architecture']):
    # ResNet18-2:
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    agent_config['cnn_encoder_feature_dim'] = 512
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResNet' in agent_config['architecture'] and 'MHDPA' in agent_config['architecture']:
    # ResNet18MHDPA-2:
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    agent_config['cnn_encoder_feature_dim'] = 512
    agent_config['mhdpa_nbr_head'] = 4
    agent_config['mhdpa_nbr_rec_update'] = 1
    agent_config['mhdpa_nbr_mlp_unit'] = 256
    agent_config['mhdpa_interaction_dim'] = 128
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = 64#512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'CNN-MHDPA' in agent_config['architecture']:
    agent_config['cnn_encoder_channels'] = [32,32,64,128]
    agent_config['cnn_encoder_kernels'] = [3,3,3,3]
    agent_config['cnn_encoder_strides'] = [1,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['mhdpa_nbr_head'] = 4
    agent_config['mhdpa_nbr_rec_update'] = 1
    agent_config['mhdpa_nbr_mlp_unit'] = 256
    agent_config['mhdpa_interaction_dim'] = 128
    agent_config['cnn_encoder_feature_dim'] = 256
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = 256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1

  # # Basic Agents

  # ## Obverter Speaker:

  # In[4]:


  from ReferentialGym.agents import DifferentiableObverterAgent


  # In[5]:


  batch_size = 4
  nbr_distractors = agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  bspeaker = DifferentiableObverterAgent(kwargs=agent_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='os0',
                                logger=logger,
                                use_sentences_one_hot_vectors=('informed' in rg_config['graphtype']))

  print("Speaker:",bspeaker)

  # ## Obverter Listener:

  # In[7]:

  batch_size = 4
  nbr_distractors = agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  blistener = DifferentiableObverterAgent(kwargs=agent_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='ol0',
                                logger=logger,
                                use_sentences_one_hot_vectors=('informed' in rg_config['graphtype']))

  print("Listener:",blistener)

  # # Dataset:

  # In[10]:

  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=rg_config['stimulus_resize_dim'], normalize_rgb_values=False)
  
  '''
  dataset_args = {
      "dataset_class":            None,
      "nbr_stimulus":             rg_config['nbr_stimulus'],
      "distractor_sampling":      rg_config['distractor_sampling'],
      "nbr_distractors":          rg_config['nbr_distractors'],
      "observability":            rg_config['observability'],
      "object_centric":           rg_config['object_centric'],
      "descriptive":              rg_config['descriptive'],
      "descriptive_target_ratio": rg_config['descriptive_target_ratio']
  }

  train_dataset = ReferentialGym.datasets.MineRLDataset(kwargs=dataset_args, root='./datasets/MineRL/', train=True, transform=transform, download=True, skip_interval=skip_interval)
  test_dataset = ReferentialGym.datasets.MineRLDataset(kwargs=dataset_args, root='./datasets/MineRL/', train=False, transform=transform, download=True, skip_interval=skip_interval)

  dataset_args['train_dataset'] = train_dataset
  dataset_args['test_dataset'] = test_dataset
  
  '''

  '''
  nbrSampledQstPerImg = 1#5
  train_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root='./datasets/Sort-of-CLEVR/', train=True, transform=transform, generate=True, nbrSampledQstPerImg=nbrSampledQstPerImg)
  test_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root='./datasets/Sort-of-CLEVR/', train=False, transform=transform, generate=True, nbrSampledQstPerImg=1)
  
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
  '''
  
  split_strategy = f"divider-{dsprites_divider}-offset-{dsprites_offset}"
  train_dataset = ReferentialGym.datasets.dSpritesDataset(root='./datasets/dsprites-dataset/', 
                                                          train=True, 
                                                          transform=transform, 
                                                          split_strategy=split_strategy)
  test_dataset = ReferentialGym.datasets.dSpritesDataset(root='./datasets/dsprites-dataset/', 
                                                         train=False, 
                                                         transform=transform, 
                                                         split_strategy=split_strategy)
  
  train_dataset[0]

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
  
  
  '''
  train_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST/', train=True, transform=transform, download=True)
  test_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST/', train=False, transform=transform, download=True)
  
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
  '''

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)

  # In[22]:

  nbr_epoch = 40
  refgame.train(prototype_speaker=bspeaker, 
                prototype_listener=blistener, 
                nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

if __name__ == '__main__':
    test_example_cultural_obverter_agents()
