#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np 
import ReferentialGym

import torch
import torchvision
import torchvision.transforms as T 

def main():
  seed = 20 #30
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  # In[23]:
  nbr_epoch = 20 #100
  cnn_feature_size = 512 # 128 512 #1024
  stimulus_resize_dim = 32 #64 #28
  normalize_rgb_values = False 
  rgb_scaler = 1.0 #255.0
  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=stimulus_resize_dim, 
                              normalize_rgb_values=normalize_rgb_values,
                              rgb_scaler=rgb_scaler)

  transform_degrees = 45
  transform_translate = (0.25, 0.25)

  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      10, #5,
      "nbr_communication_round":  1,
      "nbr_distractors":          127,
      "distractor_sampling":      "uniform",#"similarity-0.98",#"similarity-0.75",
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  
      
      "descriptive":              False,
      "descriptive_target_ratio": 0.97, 
      # Default: 1-(1/(nbr_distractors+2)), 
      # otherwise the agent find the local minimum
      # where it only predicts 'no-target'...

      "object_centric":           False,
      "nbr_stimulus":             1,

      "graphtype":                'straight_through_gumbel_softmax', #'reinforce'/'gumbel_softmax'/'straight_through_gumbel_softmax' 
      "tau0":                     0.2,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               100,
      "symbol_embedding_size":    256, #64

      "agent_architecture":       'pretrained-ResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
      "agent_learning":           'learning',  #'transfer_learning' : CNN's outputs are detached from the graph...
      "agent_loss_type":          'Hinge', #'NLL'

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

      "batch_size":               32, #128, #64
      "dataloader_num_worker":    4,
      "stimulus_depth_dim":       1,
      "stimulus_depth_mult":      1,
      "stimulus_resize_dim":      stimulus_resize_dim, 
      
      "learning_rate":            3e-4, #1e-3,
      "adam_eps":                 1e-8,
      "dropout_prob":             0.5,
      "embedding_dropout_prob":   0.8,
      
      "with_gradient_clip":       False,
      "gradient_clip":            1e0,
      
      "use_homoscedastic_multitasks_loss": False,

      "use_curriculum_nbr_distractors": False,
      "curriculum_distractors_window_size": 25, #100,

      "unsupervised_segmentation_factor": None, #1e5
      "nbr_experience_repetition":  1,

      "with_utterance_penalization":  False,
      "with_utterance_promotion":     False,
      "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
                                                # The greater this value, the greater the loss/cost.
      "utterance_factor":    1e-2,

      "with_speaker_entropy_regularization":  False,
      "with_listener_entropy_regularization":  False,
      "entropy_regularization_factor":    -1e-2,

      "with_mdl_principle":       False,
      "mdl_principle_factor":     5e-2,

      "with_weight_maxl1_loss":   False,

      "with_grad_logging":        False,
      "use_cuda":                 True,
  
      # "train_transform":          T.Compose([T.RandomAffine(degrees=transform_degrees, 
      #                                                       translate=transform_translate, 
      #                                                       scale=None, 
      #                                                       shear=None, 
      #                                                       resample=False, 
      #                                                       fillcolor=0),
      #                                         transform]),

      # "test_transform":           T.Compose([T.RandomAffine(degrees=transform_degrees, 
      #                                                      translate=transform_translate, 
      #                                                      scale=None, 
      #                                                      shear=None, 
      #                                                      resample=False, 
      #                                                      fillcolor=0),
      #                                         transform]),
  
      "train_transform":            transform,
      "test_transform":             transform,
  }

  # Normal:
  #train_split_strategy = 'combinatorial-Y-1-5-X-1-5-Orientation-1-5-Scale-1-6-Shape-1-3'
  # Aggressive:
  #train_split_strategy = 'combinatorial-Y-8-4-X-8-4-Orientation-4-5-Scale-1-5-Shape-1-3'
  train_split_strategy = 'combinatorial3-Y-4-2-X-4-2-Orientation-10-N-Scale-2-N-0FP_Shape-1-N'
  test_split_strategy = train_split_strategy
  
  '''
  train_split_strategy = 'divider-600-offset-0'
  test_split_strategy = 'divider-600-offset-50'
  '''
  '''
  train_split_strategy = 'divider-300-offset-0'
  test_split_strategy = 'divider-300-offset-25'
  '''
  '''
  train_split_strategy = 'divider-60-offset-0'
  test_split_strategy = 'divider-60-offset-25'
  '''
  
  #save_path = f"./Havrylov_et_al/test/TrainNOTF_TestNOTF/SpLayerNormOnFeatures+NoLsBatchNormOnRNN"
  #save_path = f"./Havrylov_et_al/test_Stop0Start0/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}"
  save_path = f"./Havrylov_et_al/test_Stop0Start0/PAPER/dSPrites/TestForLatentIndices/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}"
  save_path += f"/TrainNOTF_TestNOTF/SpBatchNormLsBatchNormOnFeatures+NOLsBatchNormOnRNN"
  #save_path = f"./Havrylov_et_al/test/{nbr_epoch}Ep/TrainTF_TestNOTF/SpBatchNormLsBatchNormOnFeatures+NoLsBatchNormOnRNN"
  #save_path = f"./Havrylov_et_al/test/{nbr_epoch}Ep/TrainNOTF_TestNOTF/SpLayerNormLsLayerNormOnFeatures+NoLsBatchNormOnRNN"
  save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
  save_path += f"_BN_{rg_config['agent_learning']}/"
  save_path += f"{rg_config['agent_loss_type']}/dSprites-{test_split_strategy}/OBS{rg_config['stimulus_resize_dim']}X{rg_config['stimulus_depth_dim']*rg_config['stimulus_depth_mult']}C"
  
  if rg_config['use_curriculum_nbr_distractors']:
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
  
  save_path += '-{}{}CulturalDiffObverter{}-{}GPR-SEED{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}Distr{}-stim{}-vocab{}over{}_{}{}'.\
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

  if 'MONet' in rg_config['agent_architecture'] or 'BetaVAE' in rg_config['agent_architecture']:
    save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in rg_config['agent_architecture'] else ''
    save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
    save_path += f"UnsupSeg{rg_config['unsupervised_segmentation_factor']}Rep{rg_config['nbr_experience_repetition']}" if rg_config['unsupervised_segmentation_factor'] is not None else ''
    
  rg_config['save_path'] = save_path

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agent Configuration:

  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['homoscedastic_multitasks_loss'] = rg_config['use_homoscedastic_multitasks_loss']
  agent_config['max_sentence_length'] = rg_config['max_sentence_length']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors'] if rg_config['observability'] == 'full' else 0
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['use_obverter_threshold_to_stop_message_generation'] = True
  agent_config['descriptive'] = rg_config['descriptive']
  agent_config['gumbel_softmax_eps'] = rg_config['gumbel_softmax_eps']
  agent_config['agent_learning'] = rg_config['agent_learning']

  agent_config['symbol_embedding_size'] = rg_config['symbol_embedding_size']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['dropout_prob'] = rg_config['dropout_prob']
  agent_config['embedding_dropout_prob'] = rg_config['embedding_dropout_prob']
  if 'CNN' == agent_config['architecture']:
    agent_config['cnn_encoder_channels'] = [32,32,64] #[32,32,32,32]
    agent_config['cnn_encoder_kernels'] = [8,4,3]
    agent_config['cnn_encoder_strides'] = [2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1]
    agent_config['cnn_encoder_feature_dim'] = 512 #256 #32
    agent_config['cnn_encoder_mini_batch_size'] = 256
    agent_config['temporal_encoder_nbr_hidden_units'] = 256#64#256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 256 #32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  if 'VGG16' in agent_config['architecture']:
    agent_config['cnn_encoder_feature_dim'] = 512 #256 #32
    agent_config['cnn_encoder_mini_batch_size'] = 256
    agent_config['temporal_encoder_nbr_hidden_units'] = 512#64#256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 256 #32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResNet' in agent_config['architecture']:
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size #128 #512
    agent_config['cnn_encoder_mini_batch_size'] = 32
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  # # Agents
  from ReferentialGym.agents import LSTMCNNSpeaker

  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim']*rg_config['stimulus_depth_mult'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  speaker = LSTMCNNSpeaker(kwargs=agent_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='s0',
                                logger=logger)

  print("Speaker:", speaker)

  from ReferentialGym.agents import LSTMCNNListener
  import copy

  listener_config = copy.deepcopy(agent_config)
  listener_config['nbr_distractors'] = rg_config['nbr_distractors']

  batch_size = 4
  nbr_distractors = listener_config['nbr_distractors']
  nbr_stimulus = listener_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim']*rg_config['stimulus_depth_mult'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  listener = LSTMCNNListener(kwargs=listener_config, 
                                obs_shape=obs_shape, 
                                vocab_size=vocab_size, 
                                max_sentence_length=max_sentence_length,
                                agent_id='l0',
                                logger=logger)

  print("Listener:", listener)

  # # Dataset:

  root = './datasets/dsprites-dataset'
  train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
  test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)

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

  refgame.train(prototype_speaker=speaker, 
                prototype_listener=listener, 
                nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

if __name__ == '__main__':
    main()
