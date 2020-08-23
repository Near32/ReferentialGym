import sys
import random
import numpy as np 
import argparse 
import copy

import ReferentialGym

import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision
import torchvision.transforms as T 


def main():
  parser = argparse.ArgumentParser(description='LSTM CNN Agents: ST-GS Language Emergence for Relational Reasoning.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument("--parent_folder", type=str, help="folder to save into.",default="")
  parser.add_argument('--use_cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, 
    choices=['SQOOT',
             'dSprites',
             ], 
    help='dataset to train on.',
    default='SQOOT')
  parser.add_argument('--nb_train_rhs', type=int, default=18)
  parser.add_argument('--nb_sqoot_samples', type=float, default=1e3)
  parser.add_argument('--nb_sqoot_objects', type=int, default=5)
  parser.add_argument('--nb_sqoot_shapes', type=int, default=36)
  parser.add_argument('--arch', type=str, 
    choices=["CNN",
             "CNN3x3",
             "BN+CNN",
             "BN+CNN3x3",
             "BN+3xCNN3x3",
             "BN+BetaVAE3x3",
             "BN+Coord2CNN3x3",
             "BN+Coord4CNN3x3",
             'Santoro2017-SoC-CNN',
             'Santoro2017-CLEVR-CNN',
             'Santoro2017-CLEVR-CNN3x3',
             'Santoro2017-CLEVR-EntityPrioredCNN3x3',
             'Santoro2017-CLEVR-CNN7x4x4x3',
             ], 
    help='model architecture to train',
    default="BN+CNN3x3")
    #default="Santoro2017-CLEVR-CNN")
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
  parser.add_argument('--max_sentence_length', type=int, default=20)
  parser.add_argument('--vocab_size', type=int, default=100)
  parser.add_argument('--optimizer_type', type=str, 
    choices=["adam",
             "sgd"],
    default="adam")
  parser.add_argument("--agent_loss_type", type=str,
    choices=[
      "Hinge",
      "NLL",
      "CE",
      ],
    default="Hinge")
  parser.add_argument("--agent_type", type=str,
    choices=[
      "Baseline",
      "EoSPriored",
      ],
    default="Baseline")
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--epoch', type=int, default=2000)
  parser.add_argument("--metric_epoch_period", type=int, default=20)
  parser.add_argument("--dataloader_num_worker", type=int, default=4)
  parser.add_argument("--metric_fast", action="store_true", default=False)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--mini_batch_size', type=int, default=128)
  parser.add_argument('--dropout_prob', type=float, default=0.0)
  parser.add_argument('--gumbel_softmax_tau0', type=float, default=0.2)
  parser.add_argument('--emb_dropout_prob', type=float, default=0.8)
  parser.add_argument('--nbr_experience_repetition', type=int, default=1)
  parser.add_argument('--nbr_train_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_dataset_repetition', type=int, default=1)
  parser.add_argument('--nbr_test_distractors', type=int, default=63)
  parser.add_argument('--nbr_train_distractors', type=int, default=47)
  parser.add_argument('--resizeDim', default=32, type=int,help='input image resize')
  
  parser.add_argument("--with_baseline", action="store_true", default=False)
  parser.add_argument("--from_utterances", action="store_true", default=False)
  
  parser.add_argument('--shared_architecture', action='store_true', default=False)
  parser.add_argument('--same_head', action='store_true', default=False)
  parser.add_argument('--homoscedastic_multitasks_loss', action='store_true', default=False)
  parser.add_argument("--use_curriculum_nbr_distractors", action="store_true", default=False)
  parser.add_argument('--use_feat_converter', action='store_true', default=False)
  parser.add_argument("--descriptive", action="store_true", default=False)
  parser.add_argument("--descriptive_ratio", type=float, default=0.0)
  parser.add_argument("--object_centric", action="store_true", default=False)
  parser.add_argument("--egocentric", action="store_true", default=False)
  parser.add_argument("--egocentric_tr_degrees", type=int, default=25)
  parser.add_argument("--egocentric_tr_xy", type=float, default=0.0625)
  parser.add_argument('--attached_heads', action='store_true', default=False)
  parser.add_argument('--test_id_analogy', action='store_true', default=False)
  parser.add_argument('--distractor_sampling', type=str,
    choices=[ "uniform",
              "similarity-0.98",
              "similarity-0.90",
              "similarity-0.75",
              ],
    default="uniform")
  # Obverter Hyperparameters:
  parser.add_argument('--use_sentences_one_hot_vectors', action='store_true', default=False)
  parser.add_argument('--differentiable', action='store_true', default=False)
  parser.add_argument('--obverter_threshold_to_stop_message_generation', type=float, default=0.95)
  parser.add_argument('--obverter_nbr_games_per_round', type=int, default=20)
  # Cultural Bottleneck:
  parser.add_argument('--iterated_learning_scheme', action='store_true', default=False)
  parser.add_argument('--iterated_learning_period', type=int, default=4)
  parser.add_argument('--iterated_learning_rehearse_MDL', action='store_true', default=False)
  parser.add_argument('--iterated_learning_rehearse_MDL_factor', type=float, default=1.0)
  
  # Dataset Hyperparameters:
  parser.add_argument('--train_test_split_strategy', type=str, 
    choices=['combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N', # Exp : DoRGsFurtherDise interweaved split simple XY normal             
             'combinatorial2-Y-2-S8-X-2-S8-Orientation-40-N-Scale-4-N-Shape-1-N',
             'combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N',  #Sparse: 64 imgs, 48 train, 16 test
             'combinatorial2-Y-2-S8-X-2-S8-Orientation-40-N-Scale-6-N-Shape-3-N',  # 4x Denser: 256 imgs, 192 train, 64 test,
             None,
            ],
    help='train/test split strategy',
    default=None)
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
  
  parser.add_argument('--vae_nbr_latent_dim', type=int, default=128)
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

  if args.object_centric:
    assert args.egocentric

  #if args.from_utterances:
  #  assert not args.with_baseline

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
  
  cnn_feature_size = -1 #512
  # Except for VAEs...!
  
  stimulus_resize_dim = args.resizeDim #64 #28
  
  normalize_rgb_values = False 
  
  rgb_scaler = 1.0 #255.0
  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=stimulus_resize_dim, 
                              normalize_rgb_values=normalize_rgb_values,
                              rgb_scaler=rgb_scaler)

  from ReferentialGym.datasets.utils import AddEgocentricInvariance
  ego_inv_transform = AddEgocentricInvariance()

  transform_degrees = args.egocentric_tr_degrees
  transform_translate = (args.egocentric_tr_xy, args.egocentric_tr_xy)

  default_descriptive_ratio = 1-(1/(args.nbr_train_distractors+2))
  # Default: 1-(1/(nbr_distractors+2)), 
  # otherwise the agent find the local minimum
  # where it only predicts "no-target"...
  if args.descriptive_ratio <=0.001:
    descriptive_ratio = default_descriptive_ratio
  else:
    descriptive_ratio = args.descriptive_ratio

  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      args.max_sentence_length, #5,
      "nbr_communication_round":  1,
      "nbr_distractors":          {'train':args.nbr_train_distractors, 'test':args.nbr_test_distractors},
      "distractor_sampling":      args.distractor_sampling,
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  
      
      "descriptive":              args.descriptive,
      "descriptive_target_ratio": descriptive_ratio,

      "object_centric":           args.object_centric,
      "nbr_stimulus":             1,

      "graphtype":                args.graphtype,
      "tau0":                     args.gumbel_softmax_tau0,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               args.vocab_size,
      "symbol_embedding_size":    256, #64

      "agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
      "agent_learning":           'learning',  #'transfer_learning' : CNN's outputs are detached from the graph...
      "agent_loss_type":          args.agent_loss_type,

      "cultural_pressure_it_period": None,
      "cultural_speaker_substrate_size":  1,
      "cultural_listener_substrate_size":  1,
      "cultural_reset_strategy":  "oldestL", # "uniformSL" #"meta-oldestL-SGD"
      "cultural_reset_meta_learning_rate":  1e-3,

      # Obverter's Cultural Bottleneck:
      "iterated_learning_scheme": args.iterated_learning_scheme,
      "iterated_learning_period": args.iterated_learning_period,
      "iterated_learning_rehearse_MDL": args.iterated_learning_rehearse_MDL,
      "iterated_learning_rehearse_MDL_factor": args.iterated_learning_rehearse_MDL_factor,
      
      "obverter_stop_threshold":  args.obverter_threshold_to_stop_message_generation,  #0.0 if not in use.
      "obverter_nbr_games_per_round": args.obverter_nbr_games_per_round,

      "obverter_least_effort_loss": False,
      "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

      "batch_size":               args.batch_size,
      "dataloader_num_worker":    args.dataloader_num_worker,
      "stimulus_depth_dim":       1 if 'dSprites' in args.dataset else 3,
      "stimulus_resize_dim":      stimulus_resize_dim, 
      
      "learning_rate":            args.lr, #1e-3,
      "adam_eps":                 1e-8,
      "dropout_prob":             args.dropout_prob,
      "embedding_dropout_prob":   args.emb_dropout_prob,
      
      "with_gradient_clip":       False,
      "gradient_clip":            1e0,
      
      "use_homoscedastic_multitasks_loss": args.homoscedastic_multitasks_loss,

      "use_feat_converter":       args.use_feat_converter,

      "use_curriculum_nbr_distractors": args.use_curriculum_nbr_distractors,
      "curriculum_distractors_window_size": 25, #100,

      "unsupervised_segmentation_factor": None, #1e5
      "nbr_experience_repetition":  args.nbr_experience_repetition,
      
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

      "use_cuda":                 args.use_cuda,
  
      "train_transform":            transform,
      "test_transform":             transform,
  }

  if args.egocentric:
    rg_config["train_transform"]= T.Compose(
      [
        ego_inv_transform,
        T.RandomAffine(degrees=transform_degrees, 
                     translate=transform_translate, 
                     scale=None, 
                     shear=None, 
                     resample=False, 
                     fillcolor=0),
        transform
      ]
    )
    rg_config["test_transform"]=  T.Compose(
      [
        ego_inv_transform,
        T.RandomAffine(degrees=transform_degrees, 
                     translate=transform_translate, 
                     scale=None, 
                     shear=None, 
                     resample=False, 
                     fillcolor=0),
        transform
      ]
    )

  ## Train set:
  train_split_strategy = args.train_test_split_strategy
  test_split_strategy = train_split_strategy
  
  ## Agent Configuration:
  agent_config = copy.deepcopy(rg_config)
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

  # Obverter:
  agent_config['use_obverter_threshold_to_stop_message_generation'] = args.obverter_threshold_to_stop_message_generation
  
  agent_config['symbol_embedding_size'] = rg_config['symbol_embedding_size']

  # Recurrent Convolutional Architecture:
  agent_config['architecture'] = rg_config['agent_architecture']
  agent_config['dropout_prob'] = rg_config['dropout_prob']
  agent_config['embedding_dropout_prob'] = rg_config['embedding_dropout_prob']
  
  if 'Santoro2017-SoC' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    #rg_config['use_feat_converter'] = True 
    #agent_config['use_feat_converter'] = True 
    
    # Otherwise, the VAE alone may be augmented:
    # This approach assumes that the VAE latent dimension size
    # is acting as a prior which is part of the comparison...
    rg_config['use_feat_converter'] = False
    agent_config['use_feat_converter'] = False
    
    agent_config['cnn_encoder_channels'] = ['BN32','BN64','BN128','BN256']
    if '3x3' in agent_config['architecture']:
      agent_config['cnn_encoder_kernels'] = [3,3,3,3]
    elif '7x4x4x3' in agent_config['architecture']:
      agent_config['cnn_encoder_kernels'] = [7,4,4,3]
    else:
      agent_config['cnn_encoder_kernels'] = [4,4,4,4]
    agent_config['cnn_encoder_strides'] = [2,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['cnn_encoder_fc_hidden_units'] = [] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    # Otherwise:
    cnn_feature_size = 100
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config['cnn_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    agent_config['temporal_encoder_nbr_hidden_units'] = rg_config['nbr_stimulus']*cnn_feature_size
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1

  elif 'Santoro2017-CLEVR' in agent_config['architecture']:
    # For a fair comparison between CNN an VAEs:
    # the CNN is augmented with one final FC layer reducing to the latent space shape.
    # Need to use feat converter too:
    #rg_config['use_feat_converter'] = True 
    #agent_config['use_feat_converter'] = True 
    
    # Otherwise, the VAE alone may be augmented:
    # This approach assumes that the VAE latent dimension size
    # is acting as a prior which is part of the comparison...
    rg_config['use_feat_converter'] = False
    agent_config['use_feat_converter'] = False
    
    agent_config['cnn_encoder_channels'] = ['BN24','BN24','BN24','BN24']
    if '3x3' in agent_config['architecture']:
      agent_config['cnn_encoder_kernels'] = [3,3,3,3]
    elif '7x4x4x3' in agent_config['architecture']:
      agent_config['cnn_encoder_kernels'] = [7,4,4,3]
    else:
      agent_config['cnn_encoder_kernels'] = [4,4,4,4]
    agent_config['cnn_encoder_strides'] = [2,2,2,2]
    agent_config['cnn_encoder_paddings'] = [1,1,1,1]
    agent_config['cnn_encoder_fc_hidden_units'] = [] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config['cnn_encoder_feature_dim'] = args.vae_nbr_latent_dim
    # Otherwise:
    cnn_feature_size = -1
    agent_config['cnn_encoder_feature_dim'] = cnn_feature_size
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config['cnn_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['feat_converter_output_size'] = cnn_feature_size

    if 'MHDPA' in agent_config['architecture']:
      agent_config['mhdpa_nbr_head'] = 4
      agent_config['mhdpa_nbr_rec_update'] = 1
      agent_config['mhdpa_nbr_mlp_unit'] = 256
      agent_config['mhdpa_interaction_dim'] = 128

    agent_config['temporal_encoder_nbr_hidden_units'] = 0
    agent_config['temporal_encoder_nbr_rnn_layers'] = 0
    agent_config['temporal_encoder_mini_batch_size'] = args.mini_batch_size
    agent_config['symbol_processing_nbr_hidden_units'] = agent_config['temporal_encoder_nbr_hidden_units']
    agent_config['symbol_processing_nbr_rnn_layers'] = 1

  elif "3xCNN" in agent_config["architecture"]:
    if "BN" in args.arch:
      agent_config["cnn_encoder_channels"] = ["BN32","BN64","BN128"]
    else:
      agent_config["cnn_encoder_channels"] = [32,64,128]
    
    if "3x3" in agent_config["architecture"]:
      agent_config["cnn_encoder_kernels"] = [3,3,3]
    elif "7x4x4x3" in agent_config["architecture"]:
      agent_config["cnn_encoder_kernels"] = [7,4,3]
    else:
      agent_config["cnn_encoder_kernels"] = [4,4,4]
    agent_config["cnn_encoder_strides"] = [2,2,2]
    agent_config["cnn_encoder_paddings"] = [1,1,1]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = 256
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = 256

    if "MHDPA" in agent_config["architecture"]:
      agent_config["mhdpa_nbr_head"] = 4
      agent_config["mhdpa_nbr_rec_update"] = 1
      agent_config["mhdpa_nbr_mlp_unit"] = 256
      agent_config["mhdpa_interaction_dim"] = 128

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = agent_config["temporal_encoder_nbr_hidden_units"]
    agent_config["symbol_processing_nbr_rnn_layers"] = 1

  elif "CNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    if "BN" in args.arch:
      agent_config["cnn_encoder_channels"] = ["BN32","BN32","BN64","BN64"]
    else:
      agent_config["cnn_encoder_channels"] = [32,32,64,64]
    
    if "3x3" in agent_config["architecture"]:
      agent_config["cnn_encoder_kernels"] = [3,3,3,3]
    elif "7x4x4x3" in agent_config["architecture"]:
      agent_config["cnn_encoder_kernels"] = [7,4,4,3]
    else:
      agent_config["cnn_encoder_kernels"] = [4,4,4,4]
    agent_config["cnn_encoder_strides"] = [2,2,2,2]
    agent_config["cnn_encoder_paddings"] = [1,1,1,1]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = cnn_feature_size
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = cnn_feature_size

    if "MHDPA" in agent_config["architecture"]:
      agent_config["mhdpa_nbr_head"] = 4
      agent_config["mhdpa_nbr_rec_update"] = 1
      agent_config["mhdpa_nbr_mlp_unit"] = 256
      agent_config["mhdpa_interaction_dim"] = 128

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = agent_config["temporal_encoder_nbr_hidden_units"]
    agent_config["symbol_processing_nbr_rnn_layers"] = 1

  else:
    raise NotImplementedError


  save_path_dataset = ''
  if 'SQOOT' in args.dataset:
    nb_nr_qs=5
    nb_r_qs=7
    nb_brq_qs=8
    
    generate = False
    random_generation = True
    img_size = 128 #64
    nb_objects = args.nb_sqoot_objects #2
    nb_shapes = args.nb_sqoot_shapes

    train_nb_rhs = args.nb_train_rhs
    assert train_nb_rhs > 0 and train_nb_rhs < nb_shapes
    nb_questions = nb_objects

    nb_question_types = 3
    question_size = nb_objects+nb_shapes+nb_question_types+max(nb_nr_qs, nb_r_qs, nb_brq_qs)
    
    fontScale= 1.0 #0.5
    thickness= 2 #1

    '''
    Exp0 : 
    The shape latent axis is defined as a primitive axis with 2 fillers.
    During training, there will never be more than 1 object whose shape
    is sampled within shape_ids [2,NB_SHAPES]. 
    Questions of the form q=X R Y are considered with X and Y being 
    either both from the training-time shape set, or at most one of
    them is from the testing-time shape set.
    During testing, there will be at least 2 objects whose shape is
    sampled within shape_ids [2,NB_SHAPES].
    Questions of the form q=X R Y are considered with X and Y being 
    from any shape set. In this context only, it is possible for X and 
    Y being both from the testing-time shape set. This is evaluating
    zero-shot compositional (spatial relational) inference abilities.
    '''
    #train_split_strategy = 'combinatorial1-Y-1-N-X-1-N-2IWP_Shape-1-N' 
    
    train_split_strategy = None
    nb_samples = int(args.nb_sqoot_samples)

    root = './datasets/SQOOT-dataset'
    if random_generation:
      root += f'RandomGeneration-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}-size{nb_samples}'
      save_path_dataset = f'RandomGeneration-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}-size{nb_samples}'
    else:
      root += f'-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}'
      save_path_dataset = f'-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}'
    


  save_path = ''
  if args.parent_folder == '':
    raise RuntimeError("Please provide --parent_folder.")
  
  save_path += args.parent_folder+'/'
  save_path += f"SQOOT-RHS{args.nb_train_rhs}"+save_path_dataset if "SQOOT" in args.dataset else f"{args.dataset}"
  if args.from_utterances:
    save_path += "+FromUtterances/"
  
  save_path += "+DualLabeled/"

  if args.with_baseline:
    save_path += "WithBaseline/"

  if args.attached_heads:
    save_path += "AttachedHeads/"
  if args.egocentric:
    save_path += f"Egocentric-Rot{args.egocentric_tr_degrees}-XY{args.egocentric_tr_xy}/"
  save_path += f"/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}to{args.vae_nbr_latent_dim}"
  if args.shared_architecture:
    save_path += "/shared_architecture"
  save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
  save_path += f"_BN_{rg_config['agent_learning']}/"
  save_path += f"{rg_config['agent_loss_type']}"
  
  if 'dSprites' in args.dataset: 
    train_test_strategy = f"-{test_split_strategy}"
    if test_split_strategy != train_split_strategy:
      train_test_strategy = f"/train_{train_split_strategy}/test_{test_split_strategy}"
    save_path += f"/dSprites{train_test_strategy}"
  
  save_path += f"/OBS{rg_config['stimulus_resize_dim']}X{rg_config['stimulus_depth_dim']}C-Rep{rg_config['nbr_experience_repetition']}"
  
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
    save_path += f"-ILM{rg_config['iterated_learning_period']}{'+RehearseMDL{}'.format(rg_config['iterated_learning_rehearse_MDL_factor']) if rg_config['iterated_learning_rehearse_MDL'] else ''}"
  
  if rg_config['with_mdl_principle']:
    save_path += '-MDL{}'.format(rg_config['mdl_principle_factor'])
  
  if rg_config['cultural_pressure_it_period'] != 'None':  
    save_path += '-S{}L{}-{}-Reset{}'.\
      format(rg_config['cultural_speaker_substrate_size'], 
      rg_config['cultural_listener_substrate_size'],
      rg_config['cultural_pressure_it_period'],
      rg_config['cultural_reset_strategy']+str(rg_config['cultural_reset_meta_learning_rate']) if 'meta' in rg_config['cultural_reset_strategy'] else rg_config['cultural_reset_strategy'])
  
  save_path += '-{}{}CulturalAgent-SEED{}-{}-obs_b{}_lr{}-{}-tau0-{}-{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}'.\
    format(
    'ObjectCentric' if rg_config['object_centric'] else '',
    'Descriptive{}'.format(rg_config['descriptive_target_ratio']) if rg_config['descriptive'] else '',
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
    f"/{'Detached' if args.vae_detached_featout else ''}beta{vae_beta}-factor{factor_vae_gamma}" if 'BetaVAE' in rg_config['agent_architecture'] else ''
  )

  if 'MONet' in rg_config['agent_architecture'] or 'BetaVAE' in rg_config['agent_architecture']:
    save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in rg_config['agent_architecture'] else ''
    save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
    save_path += f"UnsupSeg{rg_config['unsupervised_segmentation_factor']}" if rg_config['unsupervised_segmentation_factor'] is not None else ''
    save_path += f"LossVAECoeff{args.vae_lambda}_{'UseMu' if args.vae_use_mu_value else ''}"

  if rg_config['use_feat_converter']:
    save_path += f"+FEATCONV"
  
  if rg_config['use_homoscedastic_multitasks_loss']:
    save_path += '+Homo'
  
  save_path += f"/{args.optimizer_type}/"

  if 'reinforce' in args.graphtype:
    save_path += f'/REINFORCE_EntropyCoeffNeg1m3/UnnormalizedDetLearningSignalHavrylovLoss/NegPG/'

  if 'obverter' in args.graphtype:
    save_path += f"withPopulationHandlerModule/Obverter{args.obverter_threshold_to_stop_message_generation}-{args.obverter_nbr_games_per_round}GPR/DEBUG/"
  else:
    save_path += f"withPopulationHandlerModule/STGS-LSTM-CNN-Agent/"
  
  if args.same_head:
    save_path += "same_head/"

  if args.test_id_analogy:
    save_path += 'withAnalogyTest/'
  else:
    save_path += 'NoAnalogyTest/'
  
  save_path += f"Periodic{args.metric_epoch_period}TS+DISComp-{'fast-' if args.metric_fast else ''}/"#TestArchTanh/"
  
  save_path += f'DatasetRepTrain{args.nbr_train_dataset_repetition}Test{args.nbr_test_dataset_repetition}'
  
  rg_config['save_path'] = save_path
  
  print(save_path)

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agents
  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  if 'obverter' in args.graphtype:
    from ReferentialGym.agents import DifferentiableObverterAgent
    speaker = DifferentiableObverterAgent(
      kwargs=agent_config, 
      obs_shape=obs_shape, 
      vocab_size=vocab_size, 
      max_sentence_length=max_sentence_length,
      agent_id='s0',
      logger=logger,
      use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
      differentiable=args.differentiable
    )
  else:
    from ReferentialGym.agents import LSTMCNNSpeaker
    speaker = LSTMCNNSpeaker(
      kwargs=agent_config, 
      obs_shape=obs_shape, 
      vocab_size=vocab_size, 
      max_sentence_length=max_sentence_length,
      agent_id='s0',
      logger=logger
    )
  print("Speaker:", speaker)

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

  if 'obverter' in args.graphtype:
    listener = DifferentiableObverterAgent(
      kwargs=listener_config, 
      obs_shape=obs_shape, 
      vocab_size=vocab_size, 
      max_sentence_length=max_sentence_length,
      agent_id='l0',
      logger=logger,
      use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
      differentiable=args.differentiable
    )
  else:
    from ReferentialGym.agents import LSTMCNNListener
    listener = LSTMCNNListener(
      kwargs=listener_config, 
      obs_shape=obs_shape, 
      vocab_size=vocab_size, 
      max_sentence_length=max_sentence_length,
      agent_id='l0',
      logger=logger
    )
  print("Listener:", listener)

  # # Dataset:
  need_dict_wrapping = {}

  if 'dSprites' in args.dataset:
    root = './datasets/dsprites-dataset'
    train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
    test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)
  elif 'SQOOT' in args.dataset:
    train_dataset = ReferentialGym.datasets.SQOOTDataset(root=root, 
      train=True, 
      random_generation=random_generation,
      nb_samples=nb_samples,
      transform=rg_config['train_transform'],
      generate=generate,
      img_size=img_size,
      nb_objects=nb_objects,
      nb_shapes=nb_shapes,
      train_nb_rhs=train_nb_rhs,
      split_strategy=train_split_strategy,
      fontScale=fontScale,
      thickness=thickness
    )
    
    test_dataset = ReferentialGym.datasets.SQOOTDataset(root=root, 
      train=False, 
      transform=rg_config['test_transform'],
      generate=False,
      nb_samples=nb_samples,
      img_size=img_size,
      nb_objects=nb_objects,
      nb_shapes=nb_shapes,
      train_nb_rhs=train_nb_rhs,
      split_strategy=test_split_strategy,
      fontScale=fontScale,
      thickness=thickness
    )

    n_answers = train_dataset.nb_answers

  else:
    raise NotImplementedError  
  
  ## Modules:
  modules = {}

  from ReferentialGym import modules as rg_modules

  # Population:
  population_handler_id = "population_handler_0"
  population_handler_config = copy.deepcopy(rg_config)
  population_handler_config["verbose"] = False
  population_handler_stream_ids = {
    "current_speaker_streams_dict":"modules:current_speaker",
    "current_listener_streams_dict":"modules:current_listener",
    "epoch":"signals:epoch",
    "mode":"signals:mode",
    "global_it_datasample":"signals:global_it_datasample",
  }

  # Current Speaker:
  current_speaker_id = "current_speaker"

  # Current Listener:
  current_listener_id = "current_listener"

  # MHCM:
  if 'SQOOT' in args.dataset:
    # Baseline:
    if args.with_baseline:
      baseline_vm_id = f"baseline_{agent_config['architecture']}"
      baseline_vm_config = copy.deepcopy(agent_config)
      obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
      baseline_vm_config['obs_shape'] = obs_shape
      baselien_vm_input_stream_ids = {
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "mode":"signals:mode",
        "inputs":"current_dataloader:sample:speaker_experiences",
      }

    rg_agents_downstream_features_size = 256
    if args.attached_heads:
      rg_agents_downstream_features = "modules:current_speaker:ref_agent:features"
    else:
      rg_agents_downstream_features = "modules:current_speaker:ref_agent:features.detach"
    
    if args.from_utterances:
      lm_id = f"language_module"
      lm_config = {
        "use_cuda":agent_config["use_cuda"],
        "use_pack_padding":False,
        "use_sentences_one_hot_vectors":True,
        "rnn_type":"gru",

        "vocab_size":agent_config["vocab_size"],
        "symbol_embedding_size":agent_config["symbol_embedding_size"],
        "embedding_dropout_prob":agent_config["embedding_dropout_prob"],

        "symbol_processing_nbr_rnn_layers":agent_config["symbol_processing_nbr_rnn_layers"],
        "symbol_processing_nbr_hidden_units":agent_config["symbol_processing_nbr_hidden_units"],
        "processing_dropout_prob":agent_config["dropout_prob"],

      }

      lm_input_stream = "modules:current_speaker:sentences_one_hot"
      if not args.attached_heads:
        lm_input_stream = "modules:current_speaker:sentences_one_hot.detach"
        
      lm_input_stream_ids = {
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "mode":"signals:mode",
        "inputs":lm_input_stream,
      }

      assert lm_config["use_sentences_one_hot_vectors"] and "one_hot" in lm_input_stream_ids["inputs"]
      rg_agents_downstream_features = f"modules:{lm_id}:final_rnn_outputs"
      rg_agents_downstream_features_size = lm_config["symbol_processing_nbr_hidden_units"]
    

    fm_id = "flatten0"
    fm_input_stream_keys = [
      rg_agents_downstream_features,
    ]
  
    if args.with_baseline:
      fm_input_stream_keys.append(f"modules:{baseline_vm_id}:ref:features")

    # Batch shape is omitted in the definition of new shape...
    rrm_id = "reshaperepeat0"
    rrm_config = {
      'new_shape': [(1,-1), ],
      'repetition': [(nb_questions,1), ]
    }
    rrm_input_stream_keys = [
      "modules:flatten0:output_0",  # RG
    ]

    if args.with_baseline:
      rrm_input_stream_keys.append("modules:flatten0:output_1")
      rrm_config['new_shape'].append((1,-1))
      rrm_config['repetition'].append((nb_questions,1))

    # Binary Rel. Query needs to be extended...
    rrm_input_stream_keys.append("modules:flatten0:output_0")
    rrm_config['new_shape'].append((1,1,-1))
    rrm_config['repetition'].append((nb_questions,train_nb_rhs,1))    
    
    if args.with_baseline:
      rrm_input_stream_keys.append("modules:flatten0:output_1")
      rrm_config['new_shape'].append((1,1,-1))
      rrm_config['repetition'].append((nb_questions,train_nb_rhs,1))

    sqm_id = "squeeze_qas"
    sqm_config = {
      'dim': [None],
      #'inplace': True,
    }

    sqm_input_stream_keys = []
    for r_subtype_id in range(nb_r_qs):
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_relational_questions_{r_subtype_id}")
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_relational_answers_{r_subtype_id}")
    
    for nr_subtype_id in range(nb_nr_qs):
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_non_relational_questions_{nr_subtype_id}")
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_non_relational_answers_{nr_subtype_id}")

    for brq_subtype_id in range(nb_brq_qs):
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_binary_relational_query_questions_{brq_subtype_id}")
      sqm_input_stream_keys.append(f"current_dataloader:sample:speaker_binary_relational_query_answers_{brq_subtype_id}")
    
    cm_r_id = {}
    cm_r_config = {}
    cm_r_input_stream_keys = {}

    cm_nr_id = {}
    cm_nr_config = {}
    cm_nr_input_stream_keys = {}

    cm_brq_id = {}
    cm_brq_config = {}
    cm_brq_input_stream_keys = {}

    mhcm_r_id = {}
    mhcm_r_config = {}
    mhcm_r_input_stream_ids = {}

    mhcm_nr_id = {}
    mhcm_nr_config = {}
    mhcm_nr_input_stream_ids = {}

    mhcm_brq_id = {}
    mhcm_brq_config = {}
    mhcm_brq_input_stream_ids = {}

    # Baseline:
    if args.with_baseline:
      b_cm_r_id = {}
      b_cm_r_config = {}
      b_cm_r_input_stream_keys = {}

      b_cm_nr_id = {}
      b_cm_nr_config = {}
      b_cm_nr_input_stream_keys = {}

      b_cm_brq_id = {}
      b_cm_brq_config = {}
      b_cm_brq_input_stream_keys = {}

      b_mhcm_r_id = {}
      b_mhcm_r_config = {}
      b_mhcm_r_input_stream_ids = {}

      b_mhcm_nr_id = {}
      b_mhcm_nr_config = {}
      b_mhcm_nr_input_stream_ids = {}
      
      b_mhcm_brq_id = {}
      b_mhcm_brq_config = {}
      b_mhcm_brq_input_stream_ids = {}
      
    feature_size = 4111
    if args.resizeDim == 75 and 'Santoro2017-SoC' in args.arch:
      raise NotImplementedError
      feature_size = 4111
    #elif args.resizeDim == 75 and 'Santoro2017-CLEVR' in args.arch:
    elif args.resizeDim == 32 and 'CNN' in args.arch:
      if '3x3' in agent_config['architecture']:
        feature_size = rg_agents_downstream_features_size + question_size #52
      else:
        feature_size = 399
      mhcm_heads_arch = [256,'256-DP0.5',]
    else:
      raise NotImplementedError
      mhcm_heads_arch = [2000,2000,2000,2000, 2000,1000,500,100]
    
    
    mhcm_input_shape = feature_size

    #for subtype_id in range(max(nb_r_qs,nb_nr_qs)):
    for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
      if subtype_id < nb_r_qs:
        cm_r_id[subtype_id] = f"concat_relational_{subtype_id}"
        cm_r_config[subtype_id] = {
          'dim': -1,
        }
        cm_r_input_stream_keys[subtype_id] = [
          "modules:reshaperepeat0:output_0",
          f"modules:squeeze_qas:output_{2*subtype_id}", #0~2*(nb_r_qs-1):2 (answers are interweaved...)
        ]

      if subtype_id < nb_nr_qs:
        cm_nr_id[subtype_id] = f"concat_non_relational_{subtype_id}"
        cm_nr_config[subtype_id] = {
          'dim': -1,
        }
        cm_nr_input_stream_keys[subtype_id] = [
          "modules:reshaperepeat0:output_0",
          f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id}", #2*nb_r_qs~2*nb_r_qs+2*(nb_nr_qs-1):2 (answers are interweaved...)
        ]

      if subtype_id < nb_brq_qs:
        cm_brq_id[subtype_id] = f"concat_binary_relational_query_{subtype_id}"
        cm_brq_config[subtype_id] = {
          'dim': -1,
        }
        cm_brq_input_stream_keys[subtype_id] = [
          f"modules:reshaperepeat0:output_{2 if args.with_baseline else 1}",
          f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id}", #0~2*(nb_r_qs-1):2 (answers are interweaved...)
        ]

    if args.same_head:
      mhcm_r_id = f"mhcm_relational"
      mhcm_nr_id = f"mhcm_non_relational"
      mhcm_brq_id = f"mhcm_binary_relational_query"
      
      mhcm_config = {
        "loss_ids": {},
        "grouped_accuracies": {},
        "heads_output_sizes":[],
        "heads_archs":[],
        "input_shapes": [],
        "same_head": True,
        "use_cuda":args.use_cuda,
      }
      mhcm_input_stream_ids = {
        "losses_dict":"losses_dict",
        "logs_dict":"logs_dict",
        "mode":"signals:mode",
      }

      cmm_config = {
        "labels": [i for i in range(n_answers)],
        "input_labels": {"predicted_labels_0": "ReferentialGame"},
      }

      cmm_input_stream_ids = {}
      cmm_input_stream_ids[f"predicted_labels_0" ] = f"modules:mhcm:predicted_labels"
      cmm_input_stream_ids[f"groundtruth_labels_0" ] = f"modules:mhcm:groundtruth_labels"
      
      group_keys = []
      for subtype_id in range(nb_r_qs):
        loss_id = f"{mhcm_r_id}_{subtype_id}"
        key_id = f"inputs_{subtype_id}"
        mhcm_config['loss_ids'][key_id] = loss_id
        group_keys.append(key_id)
        mhcm_config['heads_output_sizes'].append(n_answers)
        mhcm_config['heads_archs'].append(mhcm_heads_arch)
        mhcm_config['input_shapes'].append(mhcm_input_shape)
        
        mhcm_input_stream_ids[f"inputs_{subtype_id}" ] = f"modules:concat_relational_{subtype_id}:output_0"
        mhcm_input_stream_ids[f"targets_{subtype_id}"] = f"modules:squeeze_qas:output_{2*subtype_id+1}"#1~2*nb_r_qs-1:2 (questions are interweaved...)
      mhcm_config['grouped_accuracies']["overall_relational"] = group_keys
  
      group_keys = []
      for subtype_id in range(nb_nr_qs):
        loss_id = f"{mhcm_nr_id}_{subtype_id}"
        key_id = f"inputs_{nb_r_qs + subtype_id}"
        mhcm_config['loss_ids'][key_id] = loss_id
        group_keys.append(key_id)
        mhcm_config['heads_output_sizes'].append(n_answers)
        mhcm_config['heads_archs'].append(mhcm_heads_arch)
        mhcm_config['input_shapes'].append(mhcm_input_shape)
        
        mhcm_input_stream_ids[key_id] = f"modules:concat_non_relational_{subtype_id}:output_0"
        mhcm_input_stream_ids[f"targets_{nb_r_qs + subtype_id}"] = f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id+1}" #1~2*nb_r_qs-1:2 (questions are interweaved...)
      mhcm_config['grouped_accuracies']["overall_non_relational"] = group_keys
      
      group_keys = []
      for subtype_id in range(nb_brq_qs):
        loss_id = f"{mhcm_brq_id}_{subtype_id}"
        key_id = f"inputs_{nb_r_qs + nb_nr_qs + subtype_id}"
        mhcm_config['loss_ids'][key_id] = loss_id
        group_keys.append(key_id)
        mhcm_config['heads_output_sizes'].append(n_answers)
        mhcm_config['heads_archs'].append(mhcm_heads_arch)
        mhcm_config['input_shapes'].append(mhcm_input_shape)
        
        mhcm_input_stream_ids[key_id] = f"modules:concat_binary_relational_query_{subtype_id}:output_0"
        mhcm_input_stream_ids[f"targets_{nb_r_qs + nb_nr_qs + subtype_id}"] = f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id+1}"#1~2*nb_r_qs-1:2 (questions are interweaved...)
      mhcm_config['grouped_accuracies']["overall_binary_relational_query"] = group_keys

    else:
      raise NotImplementedError
      for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          mhcm_r_id[subtype_id] = f"mhcm_relational_{subtype_id}"
          mhcm_r_config[subtype_id] = {
            'loss_ids': {"inputs_0":mhcm_r_id[subtype_id]},
            'heads_output_sizes':[n_answers],
            'heads_archs':[
              mhcm_heads_arch,
            ],
            'input_shapes': [mhcm_input_shape],
            'same_head': False,
            "use_cuda":args.use_cuda,
          }
          mhcm_r_input_stream_ids[subtype_id] = {
            "inputs_0":f"modules:concat_relational_{subtype_id}:output_0",
            "targets_0":f"modules:squeeze_qas:output_{2*subtype_id+1}", #1~2*nb_r_qs-1:2 (questions are interweaved...)
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
            "mode":"signals:mode",
          }

        if subtype_id < nb_nr_qs:
          mhcm_nr_id[subtype_id] = f"mhcm_non_relational_{subtype_id}"
          mhcm_nr_config[subtype_id] = {
            'loss_ids': {"inputs_0":mhcm_nr_id[subtype_id]},
            'heads_output_sizes':[n_answers],
            'heads_archs':[
              mhcm_heads_arch,
            ],
            'input_shapes': [mhcm_input_shape],
            'same_head': False,
            "use_cuda":args.use_cuda,
          }
          mhcm_nr_input_stream_ids[subtype_id] = {
            "inputs_0":f"modules:concat_non_relational_{subtype_id}:output_0",
            "targets_0":f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id+1}", #2*nb_r_qs+1~2*nb_r_qs+2*nb_nr_qs-1:2 (answers are interweaved...)
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
            "mode":"signals:mode",
          }

        if subtype_id < nb_brq_qs:
          mhcm_brq_id[subtype_id] = f"mhcm_binary_relational_query_{subtype_id}"
          mhcm_brq_config[subtype_id] = {
            'loss_ids': {"inputs_0":mhcm_brq_id[subtype_id]},
            'heads_output_sizes':[n_answers],
            'heads_archs':[
              mhcm_heads_arch,
            ],
            'input_shapes': [mhcm_input_shape],
            'same_head': False,
            "use_cuda":args.use_cuda,
          }
          mhcm_brq_input_stream_ids[subtype_id] = {
            "inputs_0":f"modules:concat_binary_relational_query_{subtype_id}:output_0",
            "targets_0":f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id+1}", #2*nb_r_qs+1~2*nb_r_qs+2*nb_nr_qs-1:2 (answers are interweaved...)
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
            "mode":"signals:mode",
          }

    # Baseline:
    if args.with_baseline:
      for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          b_cm_r_id[subtype_id] = f"baseline_concat_relational_{subtype_id}"
          b_cm_r_config[subtype_id] = {
            'dim': -1,
          }
          b_cm_r_input_stream_keys[subtype_id] = [
            "modules:reshaperepeat0:output_1",  # baseline visual features
            f"modules:squeeze_qas:output_{2*subtype_id}", #0~2*(nb_r_qs-1):2 (answers are interweaved...)
          ]

        if subtype_id < nb_nr_qs:
          b_cm_nr_id[subtype_id] = f"baseline_concat_non_relational_{subtype_id}"
          b_cm_nr_config[subtype_id] = {
            'dim': -1,
          }
          b_cm_nr_input_stream_keys[subtype_id] = [
            "modules:reshaperepeat0:output_1",  # baseline visual features
            f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id}", #2*nb_r_qs~2*nb_r_qs+2*(nb_nr_qs-1):2 (answers are interweaved...)
          ]

        if subtype_id < nb_brq_qs:
          b_cm_brq_id[subtype_id] = f"baseline_concat_binary_relational_query_{subtype_id}"
          b_cm_brq_config[subtype_id] = {
            'dim': -1,
          }
          b_cm_brq_input_stream_keys[subtype_id] = [
            "modules:reshaperepeat0:output_3",  # baseline visual features
            f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id}", #2*nb_r_qs~2*nb_r_qs+2*(nb_nr_qs-1):2 (answers are interweaved...)
          ]

      if args.same_head:
        b_mhcm_r_id = f"baseline_mhcm_relational"
        b_mhcm_nr_id = f"baseline_mhcm_non_relational"
        b_mhcm_brq_id = f"baseline_mhcm_binary_relational_query"
        
        b_mhcm_config = {
          'loss_ids': {},
          'grouped_accuracies': {},
          'heads_output_sizes':[],
          'heads_archs':[],
          'input_shapes': [],
          'same_head': True,
          "use_cuda":args.use_cuda,
        }
        b_mhcm_input_stream_ids = {
          "losses_dict":"losses_dict",
          "logs_dict":"logs_dict",
          "mode":"signals:mode",
        }

        b_cmm_config = {
          "labels": [i for i in range(n_answers)],
          "input_labels": {"predicted_labels_0": "Baseline"},
        }

        b_cmm_input_stream_ids = {}
        b_cmm_input_stream_ids[f"predicted_labels_0" ] = f"modules:baseline_mhcm:predicted_labels"
        b_cmm_input_stream_ids[f"groundtruth_labels_0" ] = f"modules:baseline_mhcm:groundtruth_labels"

        group_keys = []
        for subtype_id in range(nb_r_qs):
          loss_id = f"{b_mhcm_r_id}_{subtype_id}"
          key_id = f"inputs_{subtype_id}"
          b_mhcm_config['loss_ids'][key_id] = loss_id
          group_keys.append(key_id)
          b_mhcm_config['heads_output_sizes'].append(n_answers)
          b_mhcm_config['heads_archs'].append(mhcm_heads_arch)
          b_mhcm_config['input_shapes'].append(mhcm_input_shape)
          
          b_mhcm_input_stream_ids[key_id] = f"modules:baseline_concat_relational_{subtype_id}:output_0"
          b_mhcm_input_stream_ids[f"targets_{subtype_id}"] = f"modules:squeeze_qas:output_{2*subtype_id+1}" #1~2*nb_r_qs-1:2 (questions are interweaved...)
        b_mhcm_config['grouped_accuracies']["overall_relational_baseline"] = group_keys
        
        group_keys = []
        for subtype_id in range(nb_nr_qs):
          loss_id = f"{b_mhcm_nr_id}_{subtype_id}"
          key_id = f"inputs_{nb_r_qs + subtype_id}"
          b_mhcm_config['loss_ids'][key_id] = loss_id
          group_keys.append(key_id)
          b_mhcm_config['heads_output_sizes'].append(n_answers)
          b_mhcm_config['heads_archs'].append(mhcm_heads_arch)
          b_mhcm_config['input_shapes'].append(mhcm_input_shape)
          
          b_mhcm_input_stream_ids[key_id] = f"modules:baseline_concat_non_relational_{subtype_id}:output_0"
          b_mhcm_input_stream_ids[f"targets_{nb_r_qs + subtype_id}"] = f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id+1}" #1~2*nb_r_qs-1:2 (questions are interweaved...)
        b_mhcm_config['grouped_accuracies']["overall_non_relational_baseline"] = group_keys
        
        group_keys = []
        for subtype_id in range(nb_brq_qs):
          loss_id = f"{b_mhcm_brq_id}_{subtype_id}"
          key_id = f"inputs_{nb_r_qs + nb_nr_qs + subtype_id}"
          b_mhcm_config['loss_ids'][key_id] = loss_id
          group_keys.append(key_id)
          b_mhcm_config['heads_output_sizes'].append(n_answers)
          b_mhcm_config['heads_archs'].append(mhcm_heads_arch)
          b_mhcm_config['input_shapes'].append(mhcm_input_shape)
          
          b_mhcm_input_stream_ids[key_id] = f"modules:baseline_concat_binary_relational_query_{subtype_id}:output_0"
          b_mhcm_input_stream_ids[f"targets_{nb_r_qs + nb_nr_qs + subtype_id}"] = f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id+1}" #1~2*nb_r_qs-1:2 (questions are interweaved...)
        b_mhcm_config['grouped_accuracies']["overall_binary_relational_query_baseline"] = group_keys
        
      else:
        raise NotImplementedError
        for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
          if subtype_id < nb_r_qs:
            b_mhcm_r_id[subtype_id] = f"baseline_mhcm_relational_{subtype_id}"
            b_mhcm_r_config[subtype_id] = {
              'loss_ids': {"inputs_0":b_mhcm_r_id[subtype_id]},
              'heads_output_sizes':[n_answers],
              'heads_archs':[
                mhcm_heads_arch,
              ],
              'input_shapes': [mhcm_input_shape],
              'same_head': False,
              "use_cuda":args.use_cuda,
            }
            b_mhcm_r_input_stream_ids[subtype_id] = {
              "inputs_0":f"modules:baseline_concat_relational_{subtype_id}:output_0",
              "targets_0":f"modules:squeeze_qas:output_{2*subtype_id+1}", #1~2*nb_r_qs-1:2 (questions are interweaved...)
              "losses_dict":"losses_dict",
              "logs_dict":"logs_dict",
              "mode":"signals:mode",
            }

          if subtype_id < nb_nr_qs:
            b_mhcm_nr_id[subtype_id] = f"baseline_mhcm_non_relational_{subtype_id}"
            b_mhcm_nr_config[subtype_id] = {
              'loss_ids': {"inputs_0":b_mhcm_nr_id[subtype_id]},
              'heads_output_sizes':[n_answers],
              'heads_archs':[
                mhcm_heads_arch,
              ],
              'input_shapes': [mhcm_input_shape],
              'same_head': False,
              "use_cuda":args.use_cuda,
            }
            b_mhcm_nr_input_stream_ids[subtype_id] = {
              "inputs_0":f"modules:baseline_concat_non_relational_{subtype_id}:output_0",
              "targets_0":f"modules:squeeze_qas:output_{2*nb_r_qs+2*subtype_id+1}", #2*nb_r_qs+1~2*nb_r_qs+2*nb_nr_qs-1:2 (answers are interweaved...)
              "losses_dict":"losses_dict",
              "logs_dict":"logs_dict",
              "mode":"signals:mode",
            }

          if subtype_id < nb_brq_qs:
            b_mhcm_brq_id[subtype_id] = f"baseline_mhcm_binary_relational_query_{subtype_id}"
            b_mhcm_brq_config[subtype_id] = {
              'loss_ids': {"inputs_0":b_mhcm_brq_id[subtype_id]},
              'heads_output_sizes':[n_answers],
              'heads_archs':[
                mhcm_heads_arch,
              ],
              'input_shapes': [mhcm_input_shape],
              'same_head': False,
              "use_cuda":args.use_cuda,
            }
            b_mhcm_brq_input_stream_ids[subtype_id] = {
              "inputs_0":f"modules:baseline_concat_binary_relational_query_{subtype_id}:output_0",
              "targets_0":f"modules:squeeze_qas:output_{2*nb_r_qs+2*nb_nr_qs+2*subtype_id+1}", #2*nb_r_qs+1~2*nb_r_qs+2*nb_nr_qs-1:2 (answers are interweaved...)
              "losses_dict":"losses_dict",
              "logs_dict":"logs_dict",
              "mode":"signals:mode",
            }

  modules[population_handler_id] = rg_modules.build_PopulationHandlerModule(
      id=population_handler_id,
      prototype_speaker=speaker,
      prototype_listener=listener,
      config=population_handler_config,
      input_stream_ids=population_handler_stream_ids)

  modules[current_speaker_id] = rg_modules.CurrentAgentModule(id=current_speaker_id,role="speaker")
  modules[current_listener_id] = rg_modules.CurrentAgentModule(id=current_listener_id,role="listener")

  if 'dSprites' in args.dataset:
    pass
  elif 'SQOOT' in args.dataset:
    #Baseline :
    if args.with_baseline:
      modules[baseline_vm_id] = rg_modules.build_VisualModule(
        id=baseline_vm_id, 
        config=baseline_vm_config,
        input_stream_ids=baselien_vm_input_stream_ids)

    if args.from_utterances:
      modules[lm_id] = rg_modules.build_LanguageModule(
        id=lm_id,
        config=lm_config,
        input_stream_ids=lm_input_stream_ids)
      
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

    for subtype_id in range(max(nb_nr_qs, nb_r_qs, nb_brq_qs)):
      if subtype_id < nb_r_qs:
        modules[cm_r_id[subtype_id]] = rg_modules.build_ConcatModule(
          id=cm_r_id[subtype_id],
          config=cm_r_config[subtype_id],
          input_stream_keys=cm_r_input_stream_keys[subtype_id])
      if subtype_id < nb_nr_qs:
        modules[cm_nr_id[subtype_id]] = rg_modules.build_ConcatModule(
          id=cm_nr_id[subtype_id],
          config=cm_nr_config[subtype_id],
          input_stream_keys=cm_nr_input_stream_keys[subtype_id])
      if subtype_id < nb_brq_qs:
        modules[cm_brq_id[subtype_id]] = rg_modules.build_ConcatModule(
          id=cm_brq_id[subtype_id],
          config=cm_brq_config[subtype_id],
          input_stream_keys=cm_brq_input_stream_keys[subtype_id])

    if args.same_head:
      modules["mhcm"] = rg_modules.build_MultiHeadClassificationModule(
        id="mhcm", 
        config=mhcm_config,
        input_stream_ids=mhcm_input_stream_ids)

      modules["cmm"] = rg_modules.build_ConfusionMatrixMetricModule(
        id="cmm",
        config=cmm_config, 
        input_stream_ids=cmm_input_stream_ids)
    else:
      raise NotImplementedError
      for subtype_id in range(max(nb_nr_qs, nb_r_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          modules[mhcm_r_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
            id=mhcm_r_id[subtype_id], 
            config=mhcm_r_config[subtype_id],
            input_stream_ids=mhcm_r_input_stream_ids[subtype_id])
        if subtype_id < nb_nr_qs:
          modules[mhcm_nr_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
            id=mhcm_nr_id[subtype_id], 
            config=mhcm_nr_config[subtype_id],
            input_stream_ids=mhcm_nr_input_stream_ids[subtype_id])
        if subtype_id < nb_brq_qs:
          modules[mhcm_brq_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
            id=mhcm_brq_id[subtype_id], 
            config=mhcm_brq_config[subtype_id],
            input_stream_ids=mhcm_brq_input_stream_ids[subtype_id])

    # Baseline:
    if args.with_baseline:
      for subtype_id in range(max(nb_nr_qs, nb_r_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          modules[b_cm_r_id[subtype_id]] = rg_modules.build_ConcatModule(
            id=b_cm_r_id[subtype_id],
            config=b_cm_r_config[subtype_id],
            input_stream_keys=b_cm_r_input_stream_keys[subtype_id])
        if subtype_id < nb_nr_qs:
          modules[b_cm_nr_id[subtype_id]] = rg_modules.build_ConcatModule(
            id=b_cm_nr_id[subtype_id],
            config=b_cm_nr_config[subtype_id],
            input_stream_keys=b_cm_nr_input_stream_keys[subtype_id])
        if subtype_id < nb_brq_qs:
          modules[b_cm_brq_id[subtype_id]] = rg_modules.build_ConcatModule(
            id=b_cm_brq_id[subtype_id],
            config=b_cm_brq_config[subtype_id],
            input_stream_keys=b_cm_brq_input_stream_keys[subtype_id])
    
      if args.same_head:
        modules["baseline_mhcm"] = rg_modules.build_MultiHeadClassificationModule(
        id="baseline_mhcm", 
        config=b_mhcm_config,
        input_stream_ids=b_mhcm_input_stream_ids)


        modules["baseline_cmm"] = rg_modules.build_ConfusionMatrixMetricModule(
          id="baseline_cmm",
          config=b_cmm_config,
          input_stream_ids=b_cmm_input_stream_ids,
        )
      else:
        raise NotImplementedError
        for subtype_id in range(max(nb_nr_qs, nb_r_qs, nb_brq_qs)):
          if subtype_id < nb_r_qs:
            modules[b_mhcm_r_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
              id=b_mhcm_r_id[subtype_id], 
              config=b_mhcm_r_config[subtype_id],
              input_stream_ids=b_mhcm_r_input_stream_ids[subtype_id])
          if subtype_id < nb_nr_qs:
            modules[b_mhcm_nr_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
              id=b_mhcm_nr_id[subtype_id], 
              config=b_mhcm_nr_config[subtype_id],
              input_stream_ids=b_mhcm_nr_input_stream_ids[subtype_id])
          if subtype_id < nb_brq_qs:
            modules[b_mhcm_brq_id[subtype_id]] = rg_modules.build_MultiHeadClassificationModule(
              id=b_mhcm_brq_id[subtype_id], 
              config=b_mhcm_brq_config[subtype_id],
              input_stream_ids=b_mhcm_brq_input_stream_ids[subtype_id])
  else:
    raise NotImplementedError
    
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
    "optimizer_type":args.optimizer_type,
    "with_gradient_clip":rg_config["with_gradient_clip"],
    "adam_eps":rg_config["adam_eps"],
  }

  optim_module = rg_modules.build_OptimizationModule(
    id=optim_id,
    config=optim_config,
  )
  modules[optim_id] = optim_module

  grad_recorder_id = "grad_recorder"
  grad_recorder_module = rg_modules.build_GradRecorderModule(id=grad_recorder_id)
  modules[grad_recorder_id] = grad_recorder_module

  topo_sim_metric_id = "topo_sim_metric"
  topo_sim_metric_module = rg_modules.build_TopographicSimilarityMetricModule(id=topo_sim_metric_id,
    config = {
      "parallel_TS_computation_max_workers":16,
      "epoch_period":args.metric_epoch_period,
      "fast":args.metric_fast,
      "verbose":False,
      "vocab_size":rg_config["vocab_size"],
    }
  )
  modules[topo_sim_metric_id] = topo_sim_metric_module

  inst_coord_metric_id = "inst_coord_metric"
  inst_coord_metric_module = rg_modules.build_InstantaneousCoordinationMetricModule(id=inst_coord_metric_id,
    config = {
      "epoch_period":1,
    }
  )
  modules[inst_coord_metric_id] = inst_coord_metric_module

  if 'dSprites' in args.dataset:
    dsprites_latent_metric_id = "dsprites_latent_metric"
    dsprites_latent_metric_module = rg_modules.build_dSpritesPerLatentAccuracyMetricModule(id=dsprites_latent_metric_id,
      config = {
        "epoch_period":1,
      }
    )
    modules[dsprites_latent_metric_id] = dsprites_latent_metric_module

  speaker_factor_vae_disentanglement_metric_id = "speaker_factor_vae_disentanglement_metric"
  speaker_factor_vae_disentanglement_metric_input_stream_ids = {
    "model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "representations":"modules:current_speaker:ref:ref_agent:features",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
    id=speaker_factor_vae_disentanglement_metric_id,
    input_stream_ids=speaker_factor_vae_disentanglement_metric_input_stream_ids,
    config = {
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":10000,#3000,
      "nbr_eval_points":5000,#2000,
      "resample":False,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_factor_vae_disentanglement_metric_id] = speaker_factor_vae_disentanglement_metric_module

  listener_factor_vae_disentanglement_metric_id = "listener_factor_vae_disentanglement_metric"
  listener_factor_vae_disentanglement_metric_input_stream_ids = {
    "model":"modules:current_listener:ref:ref_agent:cnn_encoder",
    "representations":"modules:current_listener:ref:ref_agent:features",
    "experiences":"current_dataloader:sample:listener_experiences", 
    "latent_representations":"current_dataloader:sample:listener_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:listener_exp_latents_values",
    "indices":"current_dataloader:sample:listener_indices", 
  }
  listener_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
    id=listener_factor_vae_disentanglement_metric_id,
    input_stream_ids=listener_factor_vae_disentanglement_metric_input_stream_ids,
    config = {
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":10000,#3000,
      "nbr_eval_points":5000,#2000,
      "resample":False,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[listener_factor_vae_disentanglement_metric_id] = listener_factor_vae_disentanglement_metric_module

  if args.with_baseline:
    baseline_factor_vae_disentanglement_metric_id = "baseline_factor_vae_disentanglement_metric"
    baseline_factor_vae_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{baseline_vm_id}:ref:cnn_encoder",
      "representations":f"modules:{baseline_vm_id}:ref:features",
      "experiences":"current_dataloader:sample:listener_experiences", 
      "latent_representations":"current_dataloader:sample:listener_exp_latents", 
      "latent_values_representations":"current_dataloader:sample:listener_exp_latents_values",
      "indices":"current_dataloader:sample:listener_indices", 
    }
    baseline_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
      id=baseline_factor_vae_disentanglement_metric_id,
      input_stream_ids=baseline_factor_vae_disentanglement_metric_input_stream_ids,
      config = {
        "epoch_period":args.metric_epoch_period,
        "batch_size":64,#5,
        "nbr_train_points":10000,#3000,
        "nbr_eval_points":5000,#2000,
        "resample":False,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":True,
      }
    )
    modules[baseline_factor_vae_disentanglement_metric_id] = baseline_factor_vae_disentanglement_metric_module

  logger_id = "per_epoch_logger"
  logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
  modules[logger_id] = logger_module

  pipelines['referential_game'] = [
    population_handler_id,
    current_speaker_id,
    current_listener_id
  ]

  if 'SQOOT' in args.dataset:
    # Baseline:
    if args.with_baseline:
      pipelines[baseline_vm_id] =[
        baseline_vm_id
      ]

    # Flatten and Reshape+Repeat:
    pipelines[rrm_id+"+"+sqm_id] = []
    if args.from_utterances:  pipelines[rrm_id+"+"+sqm_id].append(lm_id)
    pipelines[rrm_id+"+"+sqm_id].append(fm_id)
    pipelines[rrm_id+"+"+sqm_id].append(rrm_id)
    pipelines[rrm_id+"+"+sqm_id].append(sqm_id)

    # Compute items:
    if args.same_head:
      for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          pipelines[cm_r_id[subtype_id]] = [
            cm_r_id[subtype_id],
          ]
        if subtype_id < nb_nr_qs:
          pipelines[cm_nr_id[subtype_id]] = [
            cm_nr_id[subtype_id],
          ]
        if subtype_id < nb_brq_qs:
          pipelines[cm_brq_id[subtype_id]] = [
            cm_brq_id[subtype_id],
          ]

      pipelines["mhcm"] = [
        "mhcm",
        "cmm",
      ]

      #Baseline:
      if args.with_baseline:
        for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
          if subtype_id < nb_r_qs:
            pipelines[b_cm_r_id[subtype_id]] = [
              b_cm_r_id[subtype_id],
            ]
          if subtype_id < nb_nr_qs:
            pipelines[b_cm_nr_id[subtype_id]] = [
              b_cm_nr_id[subtype_id],
            ]
          if subtype_id < nb_brq_qs:
            pipelines[b_cm_brq_id[subtype_id]] = [
              b_cm_brq_id[subtype_id],
            ]
        
        pipelines["baseline_mhcm"] = [
          "baseline_mhcm",
          "baseline_cmm",
        ]
    else:
      raise NotImplementedError
      for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
        if subtype_id < nb_r_qs:
          pipelines[mhcm_r_id[subtype_id]] = [
            cm_r_id[subtype_id],
            mhcm_r_id[subtype_id]
          ]
        if subtype_id < nb_nr_qs:
          pipelines[mhcm_nr_id[subtype_id]] = [
            cm_nr_id[subtype_id],
            mhcm_nr_id[subtype_id]
          ]
        if subtype_id < nb_brq_qs:
          pipelines[mhcm_brq_id[subtype_id]] = [
            cm_brq_id[subtype_id],
            mhcm_brq_id[subtype_id]
          ]

      #Baseline:
      if args.with_baseline:
        for subtype_id in range(max(nb_r_qs, nb_nr_qs, nb_brq_qs)):
          if subtype_id < nb_r_qs:
            pipelines[b_mhcm_r_id[subtype_id]] = [
              b_cm_r_id[subtype_id],
              b_mhcm_r_id[subtype_id]
            ]
          if subtype_id < nb_nr_qs:
            pipelines[b_mhcm_nr_id[subtype_id]] = [
              b_cm_nr_id[subtype_id],
              b_mhcm_nr_id[subtype_id]
            ]
          if subtype_id < nb_brq_qs:
            pipelines[b_mhcm_brq_id[subtype_id]] = [
              b_cm_brq_id[subtype_id],
              b_mhcm_brq_id[subtype_id]
            ]

  
  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)
  '''
  # Add gradient recorder module for debugging purposes:
  pipelines[optim_id].append(grad_recorder_id)
  '''
  pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
  pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
  if args.with_baseline: 
    pipelines[optim_id].append(baseline_factor_vae_disentanglement_metric_id)
  pipelines[optim_id].append(topo_sim_metric_id)
  pipelines[optim_id].append(inst_coord_metric_id)
  if 'dSprites' in args.dataset:  pipelines[optim_id].append(dsprites_latent_metric_id)
  pipelines[optim_id].append(logger_id)

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
