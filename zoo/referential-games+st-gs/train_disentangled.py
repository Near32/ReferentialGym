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
  parser = argparse.ArgumentParser(description="LSTM Disentangled Agents: ST-GS Language Emergence.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--parent_folder", type=str, help="folder to save into.",default="TestDisentangled")
  parser.add_argument("--verbose", action="store_true", default=False)
  parser.add_argument("--restore", action="store_true", default=False)
  parser.add_argument("--force_eos", action="store_true", default=False)
  parser.add_argument("--use_cuda", action="store_true", default=False)
  parser.add_argument("--dataset", type=str, 
    choices=["Sort-of-CLEVR",
             "tiny-Sort-of-CLEVR",
             "XSort-of-CLEVR",
             "tiny-XSort-of-CLEVR",
             "dSprites",
             ], 
    help="dataset to train on.",
    default="dSprites")
  parser.add_argument("--arch", type=str, 
    choices=["MLP",
             "BN+MLP",
             ], 
    help="model architecture to train",
    default="BN+MLP")
  parser.add_argument("--graphtype", type=str,
    choices=["straight_through_gumbel_softmax",
             "obverter",
             "reinforce",
             "baseline_reduced_reinforce",
             "normalized_reinforce",
             "baseline_reduced_normalized_reinforce",
             "max_entr_reinforce",
             "baseline_reduced_max_entr_reinforce",
             "baseline_reduced_normalized_max_entr_reinforce",
             "argmax_reinforce",
             ],
    help="type of graph to use during training of the speaker and listener.",
    default="straight_through_gumbel_softmax")
  parser.add_argument("--max_sentence_length", type=int, default=20)
  parser.add_argument("--vocab_size", type=int, default=100)
  parser.add_argument("--optimizer_type", type=str, 
    choices=[
      "adam",
      "sgd"
      ],
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
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--epoch", type=int, default=1875)
  parser.add_argument("--metric_epoch_period", type=int, default=20)
  parser.add_argument("--dataloader_num_worker", type=int, default=4)
  parser.add_argument("--metric_fast", action="store_true", default=False)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--mini_batch_size", type=int, default=128)
  parser.add_argument("--dropout_prob", type=float, default=0.0)
  parser.add_argument("--emb_dropout_prob", type=float, default=0.8)
  parser.add_argument("--nbr_experience_repetition", type=int, default=1)
  parser.add_argument("--nbr_train_dataset_repetition", type=int, default=1)
  parser.add_argument("--nbr_test_dataset_repetition", type=int, default=1)
  parser.add_argument("--nbr_test_distractors", type=int, default=63)
  parser.add_argument("--nbr_train_distractors", type=int, default=47)
  parser.add_argument("--resizeDim", default=32, type=int,help="input image resize")
  parser.add_argument("--shared_architecture", action="store_true", default=False)
  parser.add_argument("--homoscedastic_multitasks_loss", action="store_true", default=False)
  parser.add_argument("--use_curriculum_nbr_distractors", action="store_true", default=False)
  parser.add_argument("--use_feat_converter", action="store_true", default=False)
  parser.add_argument("--descriptive", action="store_true", default=False)
  parser.add_argument("--egocentric", action="store_true", default=False)
  parser.add_argument("--distractor_sampling", type=str,
    choices=[ "uniform",
              "similarity-0.98",
              "similarity-0.90",
              "similarity-0.75",
              ],
    default="uniform")
  # Obverter Hyperparameters:
  parser.add_argument("--use_sentences_one_hot_vectors", action="store_true", default=False)
  parser.add_argument("--differentiable", action="store_true", default=False)
  parser.add_argument("--obverter_threshold_to_stop_message_generation", type=float, default=0.95)
  parser.add_argument("--obverter_nbr_games_per_round", type=int, default=4)
  # ILM:
  parser.add_argument("--iterated_learning_scheme", action="store_true", default=False)
  parser.add_argument("--iterated_learning_period", type=int, default=4)
  parser.add_argument("--iterated_learning_rehearse_MDL", action="store_true", default=False)
  parser.add_argument("--iterated_learning_rehearse_MDL_factor", type=float, default=1.0)
  
  # Cultural Bottleneck:
  parser.add_argument("--cultural_pressure_it_period", type=int, default=None)
  parser.add_argument("--cultural_speaker_substrate_size", type=int, default=1)
  parser.add_argument("--cultural_listener_substrate_size", type=int, default=1)
  parser.add_argument("--cultural_reset_strategy", type=str, default="uniformSL") #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
      
  # Dataset Hyperparameters:
  parser.add_argument("--train_test_split_strategy", type=str, 
    choices=["combinatorial2-Y-2-8-X-2-8-Orientation-40-N-Scale-6-N-Shape-3-N", # Exp : DoRGsFurtherDise interweaved split simple XY normal             
             "combinatorial2-Y-2-S8-X-2-S8-Orientation-40-N-Scale-4-N-Shape-1-N",
             "combinatorial2-Y-32-N-X-32-N-Orientation-5-S4-Scale-1-S3-Shape-3-N",  #Sparse 2 Attributes: Orient.+Scale 64 imgs, 48 train, 16 test
             "combinatorial2-Y-2-S8-X-2-S8-Orientation-40-N-Scale-6-N-Shape-3-N",  # 4x Denser 2 Attributes: 256 imgs, 192 train, 64 test,
             
             # Heart shape: interpolation:
             "combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N",  #Sparse 2 Attributes: X+Y 64 imgs, 48 train, 16 test
             "combinatorial2-Y-2-2-X-2-2-Orientation-40-N-Scale-6-N-Shape-3-N",  #Dense 2 Attributes: X+Y 256 imgs, 192 train, 64 test
             "combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-1-2-Shape-3-N", #COMB2:Sparser 4 Attributes: 264 test / 120 train
             "combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-1-2-Shape-3-N", #COMB2:Sparse 4 Attributes: 2112 test / 960 train
             "combinatorial2-Y-2-2-X-2-2-Orientation-2-2-Scale-1-2-Shape-3-N", #COMB2:Dense 4 Attributes: ? test / ? train
             "combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N",  #COMB2 Sparse: 3 Attributes: XYOrientation 256 test / 256 train
             # Heart shape: Extrapolation:
             "combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N",  #Sparse 2 Attributes: X+Y 64 imgs, 48 train, 16 test
             "combinatorial2-Y-8-S2-X-8-S2-Orientation-10-S2-Scale-1-S3-Shape-3-N", #COMB2:Sparser 4 Attributes: 264 test / 120 train
             "combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-1-S3-Shape-3-N", #COMB2:Sparse 4 Attributes: 2112 test / 960 train
             "combinatorial2-Y-2-S8-X-2-S8-Orientation-2-S10-Scale-1-S3-Shape-3-N", #COMB2:Dense 4 Attributes: ? test / ? train
             "combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N",  #COMB2 Sparse: 3 Attributes: XYOrientation 256 test / 256 train

             # Ovale shape:
             "combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-2-N", # Denser 2 Attributes X+Y X 16/ Y 16/ --> 256 test / 768 train 
             "combinatorial2-Y-8-S2-X-8-S2-Orientation-10-S2-Scale-1-S3-Shape-2-N", #COMB2:Sparser 4 Attributes: 264 test / 120 train
             "combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-1-S3-Shape-2-N", #COMB2:Sparse 4 Attributes: 2112 test / 960 train
             "combinatorial2-Y-2-S8-X-2-S8-Orientation-2-S10-Scale-1-S3-Shape-2-N", #COMB2:Dense 4 Attributes: ? test / ? train
             
             #3 Attributes: denser 2 attributes(X+Y) with the sample size of Dense 4 attributes:
             "combinatorial2-Y-1-S16-X-1-S16-Orientation-2-S10-Scale-6-N-Shape-2-N", 
  
             "combinatorial4-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-1-S3-Shape-3-N", #Sparse 4 Attributes: 192 test / 1344 train
            ],
    help="train/test split strategy",
    # INTER 2:
    #default="combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N")
    # EXTRA 2:
    #default="combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N")
    # INTER 3:
    #default="combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N")
    # EXTRA 3:
    default="combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N")
  parser.add_argument("--fast", action="store_true", default=False, 
    help="Disable the deterministic CuDNN. It is likely to make the computation faster.")
  
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  # VAE Hyperparameters:
  #--------------------------------------------------------------------------
  #--------------------------------------------------------------------------
  parser.add_argument("--vae_detached_featout", action="store_true", default=False)

  parser.add_argument("--vae_lambda", type=float, default=1.0)
  parser.add_argument("--vae_use_mu_value", action="store_true", default=False)
  
  parser.add_argument("--vae_nbr_latent_dim", type=int, default=128)
  parser.add_argument("--vae_decoder_nbr_layer", type=int, default=3)
  parser.add_argument("--vae_decoder_conv_dim", type=int, default=32)
  
  parser.add_argument("--vae_gaussian", action="store_true", default=False)
  parser.add_argument("--vae_gaussian_sigma", type=float, default=0.25)
  
  parser.add_argument("--vae_beta", type=float, default=1.0)
  parser.add_argument("--vae_factor_gamma", type=float, default=0.0)
  
  parser.add_argument("--vae_constrained_encoding", action="store_true", default=False)
  parser.add_argument("--vae_max_capacity", type=float, default=1e3)
  parser.add_argument("--vae_nbr_epoch_till_max_capacity", type=int, default=10)

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
  
  seed = args.seed 

  # Following: https://pytorch.org/docs/stable/notes/randomness.html
  torch.manual_seed(seed)
  if hasattr(torch.backends, "cudnn") and not(args.fast):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  np.random.seed(seed)
  random.seed(seed)
  # # Hyperparameters:

  nbr_epoch = args.epoch
  
  cnn_feature_size = -1 #600 #128 #256 #
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

  transform_degrees = 25
  transform_translate = (0.0625, 0.0625)

  rg_config = {
      "observability":            "partial",
      "max_sentence_length":      args.max_sentence_length,
      "nbr_communication_round":  1,
      "nbr_distractors":          {"train":args.nbr_train_distractors, "test":args.nbr_test_distractors},
      "distractor_sampling":      args.distractor_sampling,
      # Default: use 'similarity-0.5'
      # otherwise the emerging language 
      # will have very high ambiguity...
      # Speakers find the strategy of uttering
      # a word that is relevant to the class/label
      # of the target, seemingly.  
      
      "descriptive":              args.descriptive,
      "descriptive_target_ratio": 1-(1/(args.nbr_train_distractors+2)), #0.97, 
      # Default: 1-(1/(nbr_distractors+2)), 
      # otherwise the agent find the local minimum
      # where it only predicts 'no-target'...

      "object_centric":           False,
      "nbr_stimulus":             1,

      "graphtype":                args.graphtype,
      "tau0":                     0.2,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               args.vocab_size,
      "force_eos":               args.force_eos,
      "symbol_embedding_size":    256, #64

      "agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
      "agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
      "agent_loss_type":          args.agent_loss_type, #"NLL"

      "cultural_pressure_it_period": args.cultural_pressure_it_period,
      "cultural_speaker_substrate_size":  args.cultural_speaker_substrate_size,
      "cultural_listener_substrate_size":  args.cultural_listener_substrate_size,
      "cultural_reset_strategy":  args.cultural_reset_strategy, #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
      "cultural_reset_meta_learning_rate":  1e-3,

      # Obverter's Cultural Bottleneck:
      "iterated_learning_scheme": args.iterated_learning_scheme,
      "iterated_learning_period": args.iterated_learning_period,
      "iterated_learning_rehearse_MDL": args.iterated_learning_rehearse_MDL,
      "iterated_learning_rehearse_MDL_factor": args.iterated_learning_rehearse_MDL_factor,
      
      "obverter_stop_threshold":  0.95,  #0.0 if not in use.
      "obverter_nbr_games_per_round": args.obverter_nbr_games_per_round,

      "obverter_least_effort_loss": False,
      "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

      "batch_size":               args.batch_size,
      "dataloader_num_worker":    args.dataloader_num_worker,
      "stimulus_depth_dim":       1 if "dSprites" in args.dataset else 3,
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
  agent_config["use_cuda"] = rg_config["use_cuda"]
  agent_config["homoscedastic_multitasks_loss"] = rg_config["use_homoscedastic_multitasks_loss"]
  agent_config["use_feat_converter"] = rg_config["use_feat_converter"]
  agent_config["max_sentence_length"] = rg_config["max_sentence_length"]
  agent_config["nbr_distractors"] = rg_config["nbr_distractors"]["train"] if rg_config["observability"] == "full" else 0
  agent_config["nbr_stimulus"] = rg_config["nbr_stimulus"]
  agent_config["nbr_communication_round"] = rg_config["nbr_communication_round"]
  agent_config["descriptive"] = rg_config["descriptive"]
  agent_config["gumbel_softmax_eps"] = rg_config["gumbel_softmax_eps"]
  agent_config["agent_learning"] = rg_config["agent_learning"]

  # Obverter:
  agent_config["use_obverter_threshold_to_stop_message_generation"] = args.obverter_threshold_to_stop_message_generation
  
  agent_config["symbol_embedding_size"] = rg_config["symbol_embedding_size"]

  # Recurrent Convolutional Architecture:
  agent_config["architecture"] = rg_config["agent_architecture"]
  agent_config["dropout_prob"] = rg_config["dropout_prob"]
  agent_config["embedding_dropout_prob"] = rg_config["embedding_dropout_prob"]
  
  if "MLP" in agent_config["architecture"]:
    if "BN" in args.arch:
      agent_config["hidden_units"] = ["BN256","BN256",256]
    else:
      agent_config["hidden_units"] = [256, 256, 256]
    
    agent_config['non_linearities'] = [nn.LeakyReLU]

    agent_config["symbol_processing_nbr_hidden_units"] = 256
    agent_config["symbol_processing_nbr_rnn_layers"] = 1
  else:
    raise NotImplementedError


  save_path = "./"
  if args.parent_folder != '':
    save_path += args.parent_folder+'/'
  save_path += f"{args.dataset}+DualLabeled/"
  save_path += f"/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}_CNN{cnn_feature_size}to{args.vae_nbr_latent_dim}"
  if args.shared_architecture:
    save_path += "/shared_architecture"
  if args.force_eos:
    save_path += "/forced_eos"
  save_path += f"/TrainNOTF_TestNOTF/"
  save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
  save_path += f"_BN_{rg_config['agent_learning']}/"
  save_path += f"{rg_config['agent_loss_type']}"
  
  if 'dSprites' in args.dataset: 
    train_test_strategy = f"-{test_split_strategy}"
    if test_split_strategy != train_split_strategy:
      train_test_strategy = f"/train_{train_split_strategy}/test_{test_split_strategy}"
    save_path += f"/dSprites{train_test_strategy}"
  
  save_path += f"/DisentangledOBS-Rep{rg_config['nbr_experience_repetition']}"
  
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
  
  if rg_config['cultural_pressure_it_period'] != 'None' \
    or rg_config['cultural_speaker_substrate_size'] != 1 \
    or rg_config['cultural_listener_substrate_size'] != 1:  
    save_path += '-S{}L{}-{}-Reset{}'.\
      format(rg_config['cultural_speaker_substrate_size'], 
      rg_config['cultural_listener_substrate_size'],
      rg_config['cultural_pressure_it_period'],
      rg_config['cultural_reset_strategy']+str(rg_config['cultural_reset_meta_learning_rate']) if 'meta' in rg_config['cultural_reset_strategy'] else rg_config['cultural_reset_strategy'])
  
  save_path += '-{}{}CulturalAgent-SEED{}-{}-obs_b{}_minib{}_lr{}-{}-tau0-{}-{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}'.\
    format(
    'ObjectCentric' if rg_config['object_centric'] else '',
    'Descriptive{}'.format(rg_config['descriptive_target_ratio']) if rg_config['descriptive'] else '',
    seed,
    rg_config['observability'], 
    rg_config['batch_size'], 
    args.mini_batch_size,
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
    #save_path += f'/REINFORCE_EntropyCoeffNeg1m0/UnnormalizedDetLearningSignalHavrylovLoss/NegPG/ActualProbWithExpOverLogit/TokenMaskReg/'
    save_path += f'/EntropyCoeffNeg1m3/LSRLLoss/TEST/NegPG/'

  if 'obverter' in args.graphtype:
    save_path += f"withPopulationHandlerModule/Obverter{args.obverter_threshold_to_stop_message_generation}-{args.obverter_nbr_games_per_round}GPR/DEBUG/"
  else:
    save_path += f"withPopulationHandlerModule/STGS-{args.agent_type}-LSTM-Agent/"

  save_path += f"Periodic{args.metric_epoch_period}TS+DISComp-{'fast-' if args.metric_fast else ''}/"#TestArchTanh/"
  
  save_path += f'DatasetRepTrain{args.nbr_train_dataset_repetition}Test{args.nbr_test_dataset_repetition}'
  
  rg_config['save_path'] = save_path
  
  print(save_path)

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  

  ## Modules:
  modules = {}
  from ReferentialGym import modules as rg_modules


  # # Agents
  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
  nbr_stimulus = agent_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, 240]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  from ReferentialGym.modules import OneHotEncoderModule
  ohe_id = "one_hot_encoder_0"
  ohe_config = {
    "nbr_values": 40,
    "flatten":  False,
  }
  ohe_stream_ids = {
    #"exp_latents_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
    "speaker_exp_latents":"current_dataloader:sample:speaker_exp_latents",
    "listener_exp_latents":"current_dataloader:sample:listener_exp_latents",
  }
  modules[ohe_id] = rg_modules.build_OneHotEncoderModule(
    id=ohe_id,
    config=ohe_config,
    input_stream_ids=ohe_stream_ids
  )

  from ReferentialGym.networks import FCBody
  mlp_speaker_id = "FCBody_speaker"
  mlp_speaker_config = copy.deepcopy(agent_config)
  mlp_speaker_stream_ids = {
    "speaker_exp_latents":"modules:one_hot_encoder_0:speaker_exp_latents",
  }
  modules[mlp_speaker_id] = FCBody(
    state_dim = obs_shape[-1],
    id=mlp_speaker_id,
    config=mlp_speaker_config,
    input_stream_ids=mlp_speaker_stream_ids,
    use_cuda = args.use_cuda,
  )  
  print(modules[mlp_speaker_id])

  from ReferentialGym.agents import RNNSpeaker
  speaker = RNNSpeaker(
    kwargs=agent_config, 
    obs_shape=obs_shape, 
    vocab_size=vocab_size, 
    max_sentence_length=max_sentence_length,
    agent_id='s0',
    logger=logger
  )
  speaker.input_stream_ids["speaker"]["experiences"] = f"modules:{mlp_speaker_id}:speaker_exp_latents"

  print("Speaker:", speaker)

  listener_config = copy.deepcopy(agent_config)
  if args.shared_architecture:
    listener_config['cnn_encoder'] = speaker.cnn_encoder 
  listener_config['nbr_distractors'] = rg_config['nbr_distractors']['train']
  batch_size = 4
  nbr_distractors = listener_config['nbr_distractors']
  nbr_stimulus = listener_config['nbr_stimulus']
  obs_shape = [nbr_distractors+1,nbr_stimulus, 240]
  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']


  if args.shared_architecture:
    FCBODY_id = mlp_speaker_id
    modules[mlp_speaker_id].input_stream_ids.update(
      {
        "listener_exp_latents":"modules:one_hot_encoder_0:listener_exp_latents"
      }
    )
  else:
    mlp_listener_id = "FCBody_listener"
    FCBODY_id = mlp_listener_id
    mlp_listener_config = copy.deepcopy(agent_config)
    mlp_listener_stream_ids = {
      "listener_exp_latents":"modules:one_hot_encoder_0:listener_exp_latents",
    }
    modules[mlp_listener_id] = FCBody(
      state_dim=obs_shape[-1],
      id=mlp_listener_id,
      config=mlp_listener_config,
      input_stream_ids=mlp_listener_stream_ids,
      use_cuda = args.use_cuda,
    )
    print(modules[mlp_speaker_id])

  from ReferentialGym.agents import RNNListener
  listener = RNNListener(
    kwargs=listener_config, 
    obs_shape=obs_shape, 
    vocab_size=vocab_size, 
    max_sentence_length=max_sentence_length,
    agent_id='l0',
    logger=logger
  )
  listener.input_stream_ids["listener"]["experiences"] = f"modules:{FCBODY_id}:listener_exp_latents"

  print("Listener:", listener)

  # # Dataset:
  need_dict_wrapping = {}

  if 'dSprites' in args.dataset:
    root = './datasets/dsprites-dataset'
    train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
    test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)
  else:
    raise NotImplementedError
  

  # Population:
  population_handler_id = "population_handler_0"
  population_handler_config = copy.deepcopy(rg_config)
  population_handler_config["verbose"] = args.verbose
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

  modules[population_handler_id] = rg_modules.build_PopulationHandlerModule(
      id=population_handler_id,
      prototype_speaker=speaker,
      prototype_listener=listener,
      config=population_handler_config,
      input_stream_ids=population_handler_stream_ids)

  modules[current_speaker_id] = rg_modules.CurrentAgentModule(id=current_speaker_id,role="speaker")
  modules[current_listener_id] = rg_modules.CurrentAgentModule(id=current_listener_id,role="listener")
  
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

  dsprites_latent_metric_id = "dsprites_latent_metric"
  dsprites_latent_metric_module = rg_modules.build_dSpritesPerLatentAccuracyMetricModule(id=dsprites_latent_metric_id,
    config = {
      "epoch_period":1,
    }
  )
  modules[dsprites_latent_metric_id] = dsprites_latent_metric_module

  """
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
  """

  """
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
  """

  logger_id = "per_epoch_logger"
  logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
  modules[logger_id] = logger_module

  pipelines["referential_game"] = [
    ohe_id,
    mlp_speaker_id
  ]
  if not args.shared_architecture:
    pipelines["referential_game"].append(mlp_listener_id)

  pipelines["referential_game"].append(population_handler_id)
  pipelines["referential_game"].append(current_speaker_id)
  pipelines["referential_game"].append(current_listener_id)

  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)
  """
  # Add gradient recorder module for debugging purposes:
  pipelines[optim_id].append(grad_recorder_id)
  """
  """
  pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
  pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
  """
  pipelines[optim_id].append(topo_sim_metric_id)
  pipelines[optim_id].append(inst_coord_metric_id)
  pipelines[optim_id].append(dsprites_latent_metric_id)
  pipelines[optim_id].append(logger_id)

  rg_config["modules"] = modules
  rg_config["pipelines"] = pipelines


  dataset_args = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {"train": train_dataset,
                "test": test_dataset,
                },
      "need_dict_wrapping":       need_dict_wrapping,
      "nbr_stimulus":             rg_config["nbr_stimulus"],
      "distractor_sampling":      rg_config["distractor_sampling"],
      "nbr_distractors":          rg_config["nbr_distractors"],
      "observability":            rg_config["observability"],
      "object_centric":           rg_config["object_centric"],
      "descriptive":              rg_config["descriptive"],
      "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
  }

  if args.restore:
    refgame = ReferentialGym.make(
      config=rg_config, 
      dataset_args=dataset_args,
      load_path=save_path,
      save_path=save_path,
    )
  else:
    refgame = ReferentialGym.make(
      config=rg_config, 
      dataset_args=dataset_args,
      save_path=save_path,
    )

  # In[22]:

  refgame.train(nbr_epoch=nbr_epoch,
                logger=logger,
                verbose_period=1)

  logger.flush()

if __name__ == "__main__":
    main()
