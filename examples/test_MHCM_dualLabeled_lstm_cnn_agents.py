#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import random
import numpy as np 
import argparse 

import ReferentialGym

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision
import torchvision.transforms as T 


def main():
  parser = argparse.ArgumentParser(description='LSTM CNN Agents: ST-GS Language Emergence.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--use_cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, 
    choices=['dSprites',
             'CIFAR10',
             'CIFAR100',
             'Sort-of-CLEVR',
             'tiny-Sort-of-CLEVR',
             ], 
    help='dataset to train on.',
    default='dSprites')
  parser.add_argument('--arch', type=str, 
    choices=['CNN',
             'CoordCNN',
             'FCCNN',
             'NoNonLinFCCNN',
             'smallUberCNN',
             'UberCNN',
             'CoordUberCNN',
             'ResNet18AvgPooled-2',
             'ResNet18AvgPooled-3',
             'ResNet18AvgPooledMHDPA-2',
             'pretrained-ResNet18AvgPooled-2', 
             'CoordResNet18AvgPooled-2',
             'CoordResNet18AvgPooledMHDPA-2',
             'BetaVAE',
             'CoordBetaVAE',
             'ResNet18AvgPooledBetaVAE-2',
             'pretrained-ResNet18AvgPooledBetaVAE-2',
             'CoordResNet18AvgPooledBetaVAE-2'], 
    help='model architecture to train',
    default="ResNet18AvgPooled-2")
  parser.add_argument('--graphtype', type=str,
    choices=['straight_through_gumbel_softmax',
             'reinforce',
             'baseline_reduced_reinforce',
             'normalized_reinforce',
             'baseline_reduced_normalized_reinforce',
             'max_entr_reinforce',
             'baseline_reduced_normalized_max_entr_reinforce',
             'argmax_reinforce',
             'obverter'],
    help='type of graph to use during training of the speaker and listener.',
    default='straight_through_gumbel_softmax')
  parser.add_argument('--max_sentence_length', type=int, default=15)
  parser.add_argument('--vocab_size', type=int, default=25)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--nbr_train_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_distractors', type=int, default=127)
  parser.add_argument('--nbr_train_distractors', type=int, default=127)
  parser.add_argument('--resizeDim', default=32, type=int,help='input image resize')
  parser.add_argument('--shared_architecture', action='store_true', default=False)
  parser.add_argument('--homoscedastic_multitasks_loss', action='store_true', default=False)
  parser.add_argument('--use_feat_converter', action='store_true', default=False)
  parser.add_argument('--detached_heads', action='store_true', default=False)
  parser.add_argument('--test_id_analogy', action='store_true', default=False)
  parser.add_argument('--train_test_split_strategy', type=str, 
    choices=['combinatorial3-Y-4-2-X-4-2-Orientation-10-N-Scale-2-N-Shape-1-3', # Exp1 : interweaved split
             'combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp1 : interweaved split simple X Y 
             'combinatorial2-Y-2-2-X-2-2-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp1 : interweaved split simple X Y 4xDenser 
             'combinatorial3-Y-4-E4-X-4-E4-Orientation-40-N-Scale-6-N-Shape-1-3', # Exp: Jump Around Right - splitted - medium density
             'combinatorial3-Y-4-S4-X-4-S4-Orientation-10-N-Scale-2-N-Shape-1-3', # Exp1 : splitted split
             'combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp : DoRGsFurtherDise splitted split simple XY sparse
             'combinatorial2-Y-2-S8-X-2-S8-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp : DoRGsFurtherDise splitted split simple XY normal
             'combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp : DoRGsFurtherDise interweaved split simple XY normal             
            ],
    help='train/test split strategy',
    default="combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N")
  parser.add_argument('--fast', action='store_true', default=False, 
    help='Disable the deterministic CuDNN. It is likely to make the computation faster.')
  
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  # VAE Hyperparameters:
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  parser.add_argument('--vae_detached_featout', action='store_true', default=False)

  parser.add_argument('--vae_lambda', type=float, default=1.0)
  parser.add_argument('--vae_use_mu_value', action='store_true', default=False)
  
  parser.add_argument('--vae_nbr_latent_dim', type=int, default=32)
  parser.add_argument('--vae_decoder_nbr_layer', type=int, default=3)
  parser.add_argument('--vae_decoder_conv_dim', type=int, default=32)
  
  parser.add_argument('--vae_gaussian', action='store_true', default=False)
  parser.add_argument('--vae_gaussian_sigma', type=float, default=0.25)
  
  parser.add_argument('--vae_beta', type=float, default=1.0)
  parser.add_argument('--vae_factor_gamma', type=float, default=0.0)
  
  parser.add_argument('--vae_constrained_encoding', action='store_true', default=False)
  parser.add_argument('--vae_max_capacity', type=float, default=1e3)
  parser.add_argument('--vae_nbr_epoch_till_max_capacity', type=int, default=10)

  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  
  
  args = parser.parse_args()
  print(args)


  gaussian = args.vae_gaussian 
  vae_observation_sigma = args.vae_gaussian_sigma
  
  vae_beta = args.vae_beta 
  factor_vae_gamma = args.vae_factor_gamma
  
  vae_constrainedEncoding = args.vae_constrained_encoding
  maxCap = args.vae_max_capacity #1e2
  nbrepochtillmaxcap = args.vae_nbr_epoch_till_max_capacity

  monet_gamma = 5e-1
  
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  
  
  args = parser.parse_args()
  print(args)

  seed = args.seed 

  # Following: https://pytorch.org/docs/stable/notes/randomness.html
  torch.manual_seed(seed)
  if hasattr(torch.backends, 'cudnn') and not(args.fast):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  nbr_epoch = args.epoch
  
  cnn_feature_size = 512 # 128 512 #1024
  # Except for VAEs...!
  
  stimulus_resize_dim = args.resizeDim #64 #28
  
  normalize_rgb_values = False 
  
  rgb_scaler = 1.0 #255.0
  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=stimulus_resize_dim, 
                              normalize_rgb_values=normalize_rgb_values,
                              rgb_scaler=rgb_scaler)

  transform_degrees = 45
  transform_translate = (0.25, 0.25)

  multi_head_detached = args.detached_heads 

  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      args.max_sentence_length, #5,
      "nbr_communication_round":  1,
      "nbr_distractors":          {'train':args.nbr_train_distractors, 'test':args.nbr_test_distractors},
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

      "graphtype":                args.graphtype,
      "tau0":                     0.2,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               args.vocab_size,
      "symbol_embedding_size":    256, #64

      "agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
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

      "batch_size":               args.batch_size,
      "dataloader_num_worker":    4,
      "stimulus_depth_dim":       1 if 'dSprites' in args.dataset else 3,
      "stimulus_resize_dim":      stimulus_resize_dim, 
      
      "learning_rate":            args.lr, #1e-3,
      "adam_eps":                 1e-8,
      "dropout_prob":             0.5,
      "embedding_dropout_prob":   0.8,
      
      "with_gradient_clip":       False,
      "gradient_clip":            1e0,
      
      "use_homoscedastic_multitasks_loss": args.homoscedastic_multitasks_loss,

      "use_feat_converter":       args.use_feat_converter,

      "use_curriculum_nbr_distractors": False,
      "curriculum_distractors_window_size": 25, #100,

      "unsupervised_segmentation_factor": None, #1e5
      "nbr_experience_repetition":  1,
      "nbr_dataset_repetition":  {'test':args.nbr_test_dataset_repetition, 'train':args.nbr_train_dataset_repetition},

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
      "use_cuda":                 args.use_cuda,
  
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

  ## Train set:
  train_split_strategy = args.train_test_split_strategy
  test_split_strategy = train_split_strategy
  
  ## Agent Configuration:
  agent_config = dict()
  agent_config['use_cuda'] = rg_config['use_cuda']
  agent_config['homoscedastic_multitasks_loss'] = rg_config['use_homoscedastic_multitasks_loss']
  agent_config['use_feat_converter'] = rg_config['use_feat_converter']
  agent_config['max_sentence_length'] = rg_config['max_sentence_length']
  agent_config['nbr_distractors'] = rg_config['nbr_distractors']['train'] if rg_config['observability'] == 'full' else 0
  agent_config['nbr_stimulus'] = rg_config['nbr_stimulus']
  agent_config['nbr_communication_round'] = rg_config['nbr_communication_round']
  agent_config['descriptive'] = rg_config['descriptive']
  agent_config['gumbel_softmax_eps'] = rg_config['gumbel_softmax_eps']
  agent_config['agent_learning'] = rg_config['agent_learning']

  agent_config['symbol_embedding_size'] = rg_config['symbol_embedding_size']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['dropout_prob'] = rg_config['dropout_prob']
  agent_config['embedding_dropout_prob'] = rg_config['embedding_dropout_prob']
  # if 'CNN' == agent_config['architecture']:
  #   agent_config['cnn_encoder_channels'] = [32,32,64] #[32,32,32,32]
  #   agent_config['cnn_encoder_kernels'] = [8,4,3]
  #   agent_config['cnn_encoder_strides'] = [2,2,2]
  #   agent_config['cnn_encoder_paddings'] = [1,1,1]
  #   agent_config['cnn_encoder_feature_dim'] = 512 #256 #32
  #   agent_config['cnn_encoder_mini_batch_size'] = 256
  #   agent_config['temporal_encoder_nbr_hidden_units'] = 256#64#256
  #   agent_config['temporal_encoder_nbr_rnn_layers'] = 1
  #   agent_config['temporal_encoder_mini_batch_size'] = 256 #32
  #   agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
  #   agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  if 'CNN' in agent_config['architecture'] and not('Uber' in agent_config['architecture']):
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    rg_config['use_feat_converter'] = True 
    agent_config['use_feat_converter'] = True 
    
    agent_config['cnn_encoder_channels'] = ['BN32','BN32','BN32']#,'BN32']
    agent_config['cnn_encoder_kernels'] = [4,4,4]#,4]
    agent_config['cnn_encoder_strides'] = [2,2,2]#,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1]#,1]
    agent_config['cnn_encoder_fc_hidden_units'] = [cnn_feature_size,]
    #agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    agent_config['cnn_encoder_mini_batch_size'] = 32
    
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    #agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  if 'CNN' in agent_config['architecture'] and 'Uber' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    rg_config['use_feat_converter'] = True 
    agent_config['use_feat_converter'] = True 
    
    if 'small' in agent_config['architecture']: 
      agent_config['cnn_encoder_channels'] = ['Coord8',8,8,8,8,'MP']
    else:
      agent_config['cnn_encoder_channels'] = ['Coord32',32,32,32,32,'MP']
    agent_config['cnn_encoder_kernels'] = [1,1,1,3,3,'Full']
    agent_config['cnn_encoder_strides'] = [1,1,1,1,1,1]
    agent_config['cnn_encoder_paddings'] = [0,0,0,1,1,0]
    agent_config['cnn_encoder_fc_hidden_units'] = [cnn_feature_size,]
    #agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    agent_config['cnn_encoder_mini_batch_size'] = 32
    
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    #agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  elif 'CNN' in agent_config['architecture'] and 'FC' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    rg_config['use_feat_converter'] = True 
    agent_config['use_feat_converter'] = True 
    
    if 'small' in agent_config['architecture']: 
      agent_config['cnn_encoder_channels'] = ['Coord8',8,8,8,8,'MP']
    elif 'NoNonLin' in agent_config['architecture']:
      agent_config['cnn_encoder_channels'] = ['NoNonLinCoord12',
                                              'NoNonLin12',
                                              'NoNonLin12', 
                                              'NoNonLin12',
                                              'MP']
    else:
      agent_config['cnn_encoder_channels'] = ['Coord8',32,128,cnn_feature_size,'MP']
    agent_config['cnn_encoder_kernels'] = [1,8,8,16,'Full']
    agent_config['cnn_encoder_strides'] = [1,1,1,1,1,1]
    agent_config['cnn_encoder_paddings'] = [0,0,0,1,1,0]
    agent_config['cnn_encoder_fc_hidden_units'] = None
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    #agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    agent_config['cnn_encoder_mini_batch_size'] = 32
    
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    #agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  
  elif 'VGG16' in agent_config['architecture']:
    agent_config['cnn_encoder_feature_dim'] = 512 #256 #32
    agent_config['cnn_encoder_mini_batch_size'] = 256
    agent_config['temporal_encoder_nbr_hidden_units'] = 512#64#256
    agent_config['temporal_encoder_nbr_rnn_layers'] = 1
    agent_config['temporal_encoder_mini_batch_size'] = 256 #32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
  elif 'ResNet' in agent_config['architecture'] and not('BetaVAE' in agent_config['architecture']):
    agent_config['cnn_encoder_channels'] = [32, 32, 64]
    agent_config['cnn_encoder_kernels'] = [4, 3, 3]
    agent_config['cnn_encoder_strides'] = [4, 2, 1]
    agent_config['cnn_encoder_paddings'] = [0, 1, 1]
    
    #agent_config['cnn_encoder_feature_dim'] = cnn_feature_size #128 #512
    agent_config['cnn_encoder_fc_hidden_units'] = [cnn_feature_size,]
    agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    
    agent_config['feat_converter_output_size'] = cnn_feature_size

    agent_config['cnn_encoder_mini_batch_size'] = 32
    #agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1
    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128
      
  elif 'BetaVAE' in agent_config['architecture']:
    agent_config['vae_detached_featout'] = args.vae_detached_featout

    agent_config['VAE_lambda'] = args.vae_lambda
    agent_config['vae_use_mu_value'] = args.vae_use_mu_value

    agent_config['vae_nbr_latent_dim'] = args.vae_nbr_latent_dim
    agent_config['vae_decoder_nbr_layer'] = args.vae_decoder_nbr_layer
    agent_config['vae_decoder_conv_dim'] = args.vae_decoder_conv_dim
    
    # CNN architecture to use unless there is a ResNet encoder:
    agent_config['cnn_encoder_channels'] = ['BN32','BN32','BN32']#,'BN32']
    agent_config['cnn_encoder_kernels'] = [4,4,4]#,4]
    agent_config['cnn_encoder_strides'] = [2,2,2]#,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1]#,1]
    agent_config['cnn_encoder_mini_batch_size'] = 32
    
    #agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    agent_config['cnn_encoder_fc_hidden_units'] = [cnn_feature_size,]
    agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    
    agent_config['feat_converter_output_size'] = cnn_feature_size

    agent_config['vae_beta'] = args.vae_beta
    agent_config['factor_vae_gamma'] = args.vae_factor_gamma
    agent_config['vae_use_gaussian_observation_model'] = args.vae_gaussian 
    agent_config['vae_constrainedEncoding'] = args.vae_constrained_encoding
    agent_config['vae_max_capacity'] = args.vae_max_capacity
    agent_config['vae_nbr_epoch_till_max_capacity'] = args.vae_nbr_epoch_till_max_capacity
    agent_config['vae_tc_discriminator_hidden_units'] = tuple([2*agent_config['cnn_encoder_feature_dim']]*4+[2])
    
    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*agent_config['cnn_encoder_feature_dim'] #512
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = 32
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1

    





  save_path = f"./{args.dataset}+DualLabeled/{'Attached' if not(multi_head_detached) else 'Detached'}Heads"
  save_path += f"/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}to{args.vae_nbr_latent_dim}"
  if args.shared_architecture:
    save_path += "/shared_architecture"
  save_path += f"/TrainNOTF_TestNOTF/SpBatchNormLsBatchNormOnFeatures+LsBatchNormOnRNN"
  save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
  save_path += f"_BN_{rg_config['agent_learning']}/"
  save_path += f"{rg_config['agent_loss_type']}"
  
  if 'dSprites' in args.dataset: 
    train_test_strategy = f"-{test_split_strategy}"
    if test_split_strategy != train_split_strategy:
      train_test_strategy = f"/train_{train_split_strategy}/test_{test_split_strategy}"
    save_path += f"/dSprites{train_test_strategy}"
  
  save_path += f"/OBS{rg_config['stimulus_resize_dim']}X{rg_config['stimulus_depth_dim']}C"
  
  if rg_config['use_curriculum_nbr_distractors']:
    save_path += f"+W{rg_config['curriculum_distractors_window_size']}Curr"
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
  
  save_path += '-{}{}CulturalDiffObverter{}-{}GPR-SEED{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}'.\
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
    *rg_config['nbr_distractors'].values(), 
    rg_config['nbr_stimulus'], 
    rg_config['vocab_size'], 
    rg_config['max_sentence_length'], 
    rg_config['agent_architecture'],
    f"/{'Detached' if args.vae_detached_featout else ''}beta{vae_beta}-factor{factor_vae_gamma}" if 'BetaVAE' in rg_config['agent_architecture'] else '')

  if 'MONet' in rg_config['agent_architecture'] or 'BetaVAE' in rg_config['agent_architecture']:
    save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in rg_config['agent_architecture'] else ''
    save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
    save_path += f"UnsupSeg{rg_config['unsupervised_segmentation_factor']}Rep{rg_config['nbr_experience_repetition']}" if rg_config['unsupervised_segmentation_factor'] is not None else ''
    save_path += f"LossVAECoeff{args.vae_lambda}_{'UseMu' if args.vae_use_mu_value else ''}"

  if rg_config['use_feat_converter']:
    save_path += f"+FEATCONV"
  
  if rg_config['use_homoscedastic_multitasks_loss']:
    save_path += '+Homo'
  
  if 'reinforce' in args.graphtype:
    save_path += f'/REINFORCE_EntropyCoeffNeg1m3/UnnormalizedDetLearningSignalHavrylovLoss/NegPG/'

  save_path += f"withPopulationHandlerModule//STGS-LSTM-CNN-Agent/"
  if args.test_id_analogy:
    save_path += 'withAnalogyTest/'
  else:
    save_path += 'NoAnalogyTest/'
  
  save_path += f'DatasetRepTrain{args.nbr_train_dataset_repetition}Test{args.nbr_test_dataset_repetition}'
  
  rg_config['save_path'] = save_path
  
  print(save_path)

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agents
  from ReferentialGym.agents import LSTMCNNSpeaker

  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  speaker = LSTMCNNSpeaker(
    kwargs=agent_config, 
    obs_shape=obs_shape, 
    vocab_size=vocab_size, 
    max_sentence_length=max_sentence_length,
    agent_id='s0',
    logger=logger
  )

  print("Speaker:", speaker)

  from ReferentialGym.agents import LSTMCNNListener
  import copy

  listener_config = copy.deepcopy(agent_config)
  
  if args.shared_architecture:
    listener_config['cnn_encoder'] = speaker.cnn_encoder 

  listener_config['nbr_distractors'] = rg_config['nbr_distractors']['train']

  batch_size = 4
  nbr_distractors = listener_config['nbr_distractors']
  nbr_stimulus = listener_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
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
  need_dict_wrapping = {}

  if 'dSprites' in args.dataset:
    root = './datasets/dsprites-dataset'
    train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
    test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)
  elif 'CIFAR10' in args.dataset:
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/', train=True, transform=rg_config['train_transform'], download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/', train=False, transform=rg_config['test_transform'], download=True)
    need_dict_wrapping["train"] = True
    need_dict_wrapping["test"] = True
  elif 'Sort-of-CLEVR' in args.dataset:
    if 'tiny' in args.dataset:
      generate=True 
      dataset_size=1000
      test_size=200
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3
    else:
      generate=True 
      dataset_size=10000
      test_size=2000
      img_size=75
      object_size=5
      nb_objects=6
      test_id_analogy = args.test_id_analogy
      test_id_analogy_threshold = 3

    n_answers = 4+nb_objects
    if test_id_analogy:
      nb_questions = 3
    else:
      nb_questions = 6

    root = './datasets/sort-of-CLEVR-dataset'
    root += f'-{dataset_size}'
    root += f'-imgS{img_size}-objS{object_size}-obj{nb_objects}'
    
    train_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root=root, 
      train=True, 
      transform=rg_config['train_transform'],
      generate=generate,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)
    
    test_dataset = ReferentialGym.datasets.SortOfCLEVRDataset(root=root, 
      train=False, 
      transform=rg_config['test_transform'],
      generate=False,
      dataset_size=dataset_size,
      test_size=test_size,
      img_size=img_size,
      object_size=object_size,
      nb_objects=nb_objects,
      test_id_analogy=test_id_analogy,
      test_id_analogy_threshold=test_id_analogy_threshold)

  
  
  ## Modules:
  modules = {}

  from ReferentialGym import modules as rg_modules

  # Population:
  population_handler_id = "population_handler_0"
  population_handler_config = rg_config
  population_handler_stream_ids = {
    "modules:current_speaker":"current_speaker_streams_dict",
    "modules:current_listener":"current_listener_streams_dict",
    "signals:epoch":"epoch",
    "signals:mode":"mode",
    "signals:it_datasample":"it_datasample",
  }

  # Current Speaker:
  current_speaker_id = "current_speaker"

  # Current Listener:
  current_listener_id = "current_listener"


  # MHCM:
  if 'dSprites' in args.dataset:
    mhcm_ffm_id = "mhcm0"
    mhcm_ffm_config = {
      'loss_id': mhcm_ffm_id,
      'heads_output_sizes':[2, 3, 6, 40, 32, 32],
      'heads_archs':[
        [512],
      ],
      'input_stream_module': speaker.cnn_encoder,
      'detach_input': multi_head_detached,
      "use_cuda":args.use_cuda,
    }

    mhcm_ffm_input_stream_ids = {
      "modules:current_speaker:ref_agent:feat_maps":"inputs",
      "current_dataloader:sample:speaker_exp_latents":"targets",
      "losses_dict":"losses_dict",
      "logs_dict":"logs_dict",
    }
  elif 'Sort-of-CLEVR' in args.dataset:
    fm_id = "flatten0"
    if args.detached_heads:
      fm_input_stream_keys = [
        "modules:current_speaker:ref_agent:feat_maps.detach",
      ]
    else:
      fm_input_stream_keys = [
        "modules:current_speaker:ref_agent:feat_maps",
      ]

    rrm_id = "reshaperepeat0"
    rrm_config = {
      'new_shape': [(1,-1)],
      'repetition': [(nb_questions,1)]
    }
    rrm_input_stream_keys = [
      "modules:flatten0:output_0",
    ]

    sqm_id = "squeeze_qas"
    sqm_config = {
      'dim': [None],
      #'inplace': True,
    }
    sqm_input_stream_keys = [
      "current_dataloader:sample:speaker_relational_questions_0", #0
      "current_dataloader:sample:speaker_relational_questions_1", #1
      "current_dataloader:sample:speaker_relational_questions_2", #2
      "current_dataloader:sample:speaker_relational_answers_0",   #3
      "current_dataloader:sample:speaker_relational_answers_1",   #4
      "current_dataloader:sample:speaker_relational_answers_2",   #5
      "current_dataloader:sample:speaker_non_relational_questions_0", #6
      "current_dataloader:sample:speaker_non_relational_questions_1", #7
      "current_dataloader:sample:speaker_non_relational_questions_2", #8
      "current_dataloader:sample:speaker_non_relational_answers_0",   #9
      "current_dataloader:sample:speaker_non_relational_answers_1",   #10
      "current_dataloader:sample:speaker_non_relational_answers_2",   #11
    ]

    cm_r_id = {}
    cm_r_config = {}
    cm_r_input_stream_keys = {}

    cm_nr_id = {}
    cm_nr_config = {}
    cm_nr_input_stream_keys = {}

    mhcm_r_id = {}
    mhcm_r_config = {}
    mhcm_r_input_stream_ids = {}

    mhcm_nr_id = {}
    mhcm_nr_config = {}
    mhcm_nr_input_stream_ids = {}
    
    feature_size = 2059 if args.resizeDim == 32 else 4107

    for subtype_id in range(3):
      cm_r_id[subtype_id] = f"concat_relational_{subtype_id}"
      cm_r_config[subtype_id] = {
        'dim': -1,
      }
      cm_r_input_stream_keys[subtype_id] = [
        "modules:reshaperepeat0:output_0",
        f"modules:squeeze_qas:output_{subtype_id}",
      ]

      cm_nr_id[subtype_id] = f"concat_non_relational_{subtype_id}"
      cm_nr_config[subtype_id] = {
        'dim': -1,
      }
      cm_nr_input_stream_keys[subtype_id] = [
        "modules:reshaperepeat0:output_0",
        f"modules:squeeze_qas:output_{6+subtype_id}",
      ]

      mhcm_r_id[subtype_id] = f"mhcm_relational_{subtype_id}"
      mhcm_r_config[subtype_id] = {
        'loss_id': mhcm_r_id[subtype_id],
        'heads_output_sizes':[n_answers],
        'heads_archs':[
          [512],
        ],
        #'input_shape': 520 if "CNN" in args.arch else (2059 if 'tiny' in args.dataset else 2056),
        'input_shape': 520 if "CNN" in args.arch else (feature_size if 'tiny' in args.dataset else 2056),
        'detach_input': multi_head_detached,
        "use_cuda":args.use_cuda,
      }
      mhcm_r_input_stream_ids[subtype_id] = {
        f"modules:concat_relational_{subtype_id}:output_0":"inputs",
        f"modules:squeeze_qas:output_{3+subtype_id}":"targets",
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "signals:mode":"mode",
      }

      mhcm_nr_id[subtype_id] = f"mhcm_non_relational_{subtype_id}"
      mhcm_nr_config[subtype_id] = {
        'loss_id': mhcm_nr_id[subtype_id],
        'heads_output_sizes':[n_answers],
        'heads_archs':[
          [512],
        ],
        #'input_shape': 520 if "CNN" in args.arch else (2059 if 'tiny' in args.dataset else 2056),
        'input_shape': 520 if "CNN" in args.arch else (feature_size if 'tiny' in args.dataset else 2056),
        'detach_input': multi_head_detached,
        "use_cuda":args.use_cuda,
      }
      mhcm_nr_input_stream_ids[subtype_id] = {
        f"modules:concat_non_relational_{subtype_id}:output_0":"inputs",
        f"modules:squeeze_qas:output_{9+subtype_id}":"targets",
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "signals:mode":"mode",
      }

  # TODO:
  '''
  -1) Implement Network nn.Modules as Modules...
  ''' 
  '''
  0)  Implemente output_stream definition to Modules...
  '''
  
  modules[population_handler_id] = rg_modules.build_PopulationHandlerModule(
      id=population_handler_id,
      prototype_speaker=speaker,
      prototype_listener=listener,
      config=population_handler_config,
      input_stream_ids=population_handler_stream_ids)

  modules[current_speaker_id] = rg_modules.CurrentAgentModule(id=current_speaker_id,role="speaker")
  modules[current_listener_id] = rg_modules.CurrentAgentModule(id=current_listener_id,role="listener")

  if 'dSprites' in args.dataset:
    modules[mhcm_ffm_id] = rg_modules.build_MultiHeadClassificationFromFeatureMapModule(
      id=mhcm_ffm_id, 
      config=mhcm_ffm_config,
      input_stream_ids=mhcm_ffm_input_stream_ids)
  elif 'Sort-of-CLEVR' in args.dataset:
    modules[fm_id] = rg_modules.build_FlattenModule(
      id=fm_id,
      input_stream_keys=fm_input_stream_keys)
    modules[rrm_id] = rg_modules.build_BatchReshapeRepeatModule(
      id=rrm_id,
      config=rrm_config,
      input_stream_keys=rrm_input_stream_keys)
    modules[sqm_id] = rg_modules.build_SqueezeModule(
      id=sqm_id,
      config=sqm_config,
      input_stream_keys=sqm_input_stream_keys)

    for subtype_id in range(3):
      modules[cm_r_id[subtype_id]] = rg_modules.build_ConcatModule(
        id=cm_r_id[subtype_id],
        config=cm_r_config[subtype_id],
        input_stream_keys=cm_r_input_stream_keys[subtype_id])
      modules[cm_nr_id[subtype_id]] = rg_modules.build_ConcatModule(
        id=cm_nr_id[subtype_id],
        config=cm_nr_config[subtype_id],
        input_stream_keys=cm_nr_input_stream_keys[subtype_id])

      modules[mhcm_r_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
        id=mhcm_r_id[subtype_id], 
        config=mhcm_r_config[subtype_id],
        input_stream_ids=mhcm_r_input_stream_ids[subtype_id])

      modules[mhcm_nr_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
        id=mhcm_nr_id[subtype_id], 
        config=mhcm_nr_config[subtype_id],
        input_stream_ids=mhcm_nr_input_stream_ids[subtype_id])

  homo_id = "homo0"
  homo_config = {"use_cuda":args.use_cuda}
  if args.homoscedastic_multitasks_loss:
    modules[homo_id] = rg_modules.build_HomoscedasticMultiTasksLossModule(
      id=homo_id,
      config=homo_config,
    )
  
  ## Pipelines:
  pipelines = {}

  # 0) Now that all the modules are known, let us build the optimization module:
  optim_id = "global_optim"
  optim_config = {
    "modules":modules,
    "learning_rate":args.lr,
    "with_gradient_clip":rg_config["with_gradient_clip"],
    "adam_eps":rg_config["adam_eps"],
  }

  optim_module = rg_modules.build_OptimizationModule(
    id=optim_id,
    config=optim_config,
  )
  modules[optim_id] = optim_module

  pipelines['referential_game'] = [
    population_handler_id,
    current_speaker_id,
    current_listener_id
  ]

  if 'dSprites' in args.dataset:
    pipelines[mhcm_id] = [
      mhcm_id
    ]

  if 'Sort-of-CLEVR' in args.dataset:
    # Flatten and Reshape+Repeat:
    pipelines[rrm_id+"+"+sqm_id] = [
      fm_id,
      rrm_id,
      sqm_id
    ]

    # Compute relational items:
    for subtype_id in range(3):
      pipelines[mhcm_r_id[subtype_id]] = [
        cm_r_id[subtype_id],
        mhcm_r_id[subtype_id]
      ]

      # Compute non-relational items:
      pipelines[mhcm_nr_id[subtype_id]] = [
        cm_nr_id[subtype_id],
        mhcm_nr_id[subtype_id]
      ]
  
  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)

  rg_config["modules"] = modules
  rg_config["pipelines"] = pipelines


  dataset_args = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {"train": train_dataset,
                "test": test_dataset,
                },
      "need_dict_wrapping":       need_dict_wrapping,
      "nbr_stimulus":             rg_config['nbr_stimulus'],
      "distractor_sampling":      rg_config['distractor_sampling'],
      "nbr_distractors":          rg_config['nbr_distractors'],
      "observability":            rg_config['observability'],
      "object_centric":           rg_config['object_centric'],
      "descriptive":              rg_config['descriptive'],
      "descriptive_target_ratio": rg_config['descriptive_target_ratio'],
  }

  refgame = ReferentialGym.make(config=rg_config, dataset_args=dataset_args)

  # In[22]:

  refgame.train(nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

  logger.flush()

if __name__ == '__main__':
    main()
