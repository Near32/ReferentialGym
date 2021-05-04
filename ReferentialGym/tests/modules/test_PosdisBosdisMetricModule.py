from typing import Dict, List

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

import logging
logging.disable(logging.WARNING)

from dummy_dataset import DummyDataset

from ReferentialGym.modules import Module 


# python -m ipdb -c c ./RG/ReferentialGym/tests/modules/test_PosdisBosdisMetricModule.py --parent_folder ./MetricTesting/STGS/LogSM/ReLU/ObjectCentric//SymbolEmb64+GRU64+CNN256+Decision128/ --use_cuda --seed 10 --obverter_nbr_games_per_round 20 --obverter_threshold_to_stop_message_generation 0.95 --emb_dropout_prob 0.0 --dropout_prob 0.0 --use_sentences_one_hot_vectors --batch_size 100 --mini_batch_size 256 --resizeDim 64 --arch SmallBaselineCNN --max_sentence_length 20 --vocab_size 20 --epoch 30 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 --nbr_train_distractors 1 --nbr_test_distractors 1 --nbr_experience_repetition 1 --obverter_use_decision_head --agent_loss_type NLL --metric_epoch_period 1 --nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 --lr 6e-4 --metric_fast --nbr_train_points 5000 --nbr_eval_points 5000

class ProgressiveShuffleModule(Module):
  def __init__(self, id:str, config:Dict[str,object], input_stream_ids:Dict[str,str]):
    default_input_stream_ids = {
      "logs_dict":"logs_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",

      "end_of_dataset":"signals:end_of_dataset",  
      # boolean: whether the current batch/datasample is the last of the current dataset/mode.
      "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
      # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
      # is the last of the current sequence of repetition.
      "end_of_communication":"signals:end_of_communication",
      # boolean: whether the current communication round is the last of 
      # the current dialog.

      "input":"current_dataloader:sample:speaker_exp_latents", 
    }
    if input_stream_ids is None:
      input_stream_ids = default_input_stream_ids
    else:
      for default_stream, default_id in default_input_stream_ids.items():
        if default_id not in input_stream_ids.values():
          input_stream_ids[default_stream] = default_id

    super(ProgressiveShuffleModule, self).__init__(
      id=id,
      type="ProgressiveShuffleModule",
      config=config,
      input_stream_ids=input_stream_ids
    )

    self.shuffle_percentage = 0.01
    #self.shuffle_percentage = 0.5
    #self.shuffle_percentage = 1.0
    self.shuffle_period_increment = self.config["shuffle_period_increment"] # expect float ]0,1.0]
    
    self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
    self.random_state_rep = np.random.RandomState(0)
  
  def forward(self, x):
    out = torch.from_numpy(x).float()
    size = int(self.shuffle_percentage*x.shape[0])
    if size != 0:
      for idx in range(size):
        out[idx] = out[idx, ..., torch.randperm(out.shape[-1])]
    return out.numpy()

  def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object]:
    outputs_stream_dict = {}

    logs_dict = input_streams_dict["logs_dict"]
    mode = input_streams_dict["mode"]
    
    inp = input_streams_dict["input"].float()
    
    output = inp
    
    size = int(self.shuffle_percentage*inp.shape[0])
    if size != 0:
      for idx in range(size):
        output[idx] = output[idx, ..., torch.randperm(output.shape[-1])]
    
    outputs_stream_dict["output"] = output

    # Is it the end of the epoch?
    end_of_epoch = all([
      input_streams_dict[key]
      for key in self.end_of_]
    )
    
    # If so, let us average over every value and save it:
    if end_of_epoch:
      self.shuffle_percentage = min(1.0, self.shuffle_percentage+self.shuffle_period_increment)

    logs_dict[f"{mode}/{self.id}/ShufflePercentage"] = self.shuffle_percentage*100.0
    logs_dict[f"{mode}/{self.id}/ShufflePercentage/size"] = size

    return outputs_stream_dict

class ProgressiveBatchShuffleModule(Module):
  def __init__(self, id:str, config:Dict[str,object], input_stream_ids:Dict[str,str]):
    default_input_stream_ids = {
      "logs_dict":"logs_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",

      "end_of_dataset":"signals:end_of_dataset",  
      # boolean: whether the current batch/datasample is the last of the current dataset/mode.
      "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
      # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
      # is the last of the current sequence of repetition.
      "end_of_communication":"signals:end_of_communication",
      # boolean: whether the current communication round is the last of 
      # the current dialog.

      "input":"current_dataloader:sample:speaker_exp_latents", 
    }
    if input_stream_ids is None:
      input_stream_ids = default_input_stream_ids
    else:
      for default_stream, default_id in default_input_stream_ids.items():
        if default_id not in input_stream_ids.values():
          input_stream_ids[default_stream] = default_id

    super(ProgressiveBatchShuffleModule, self).__init__(
      id=id,
      type="ProgressiveBatchShuffleModule",
      config=config,
      input_stream_ids=input_stream_ids
    )

    self.shuffle_percentage = 0.01
    #self.shuffle_percentage = 0.5
    #self.shuffle_percentage = 1.0
    self.shuffle_period_increment = self.config["shuffle_period_increment"] # expect float ]0,1.0]
    
    self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
    self.random_state_rep = np.random.RandomState(0)
  
  def forward(self, x):
    out = torch.from_numpy(x).float()
    size = int(self.shuffle_percentage*x.shape[-1])
    if size != 0:
      for idx in range(size):
        out[:, ..., idx] = out[torch.randperm(out.shape[0]), ..., idx]
    return out.numpy()

  def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object]:
    outputs_stream_dict = {}

    logs_dict = input_streams_dict["logs_dict"]
    mode = input_streams_dict["mode"]
    
    inp = input_streams_dict["input"].float()
    
    output = inp
    
    size = int(self.shuffle_percentage*inp.shape[-1])
    if size != 0:
      for idx in range(size):
        output[:, ..., idx] = output[torch.randperm(output.shape[0]), ..., idx]
        
    outputs_stream_dict["output"] = output

    # Is it the end of the epoch?
    end_of_epoch = all([
      input_streams_dict[key]
      for key in self.end_of_]
    )
    
    # If so, let us average over every value and save it:
    if end_of_epoch:
      self.shuffle_percentage = min(1.0, self.shuffle_percentage+self.shuffle_period_increment)

    logs_dict[f"{mode}/{self.id}/BatchShufflePercentage"] = self.shuffle_percentage*100.0
    logs_dict[f"{mode}/{self.id}/BatchShufflePercentage/size"] = size

    return outputs_stream_dict


def main():
  parser = argparse.ArgumentParser(description="STGS Agents: Language Emergence on 3DShapesPyBullet Dataset.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--parent_folder", type=str, help="folder to save into.",default="TestObverter")
  parser.add_argument("--use_obverter_sampling", action="store_true", default=False)
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
             "DummyDataset",
             "3DShapesPyBullet",
             ], 
    help="dataset to train on.",
    default="DummyDataset")
    #default="dSprites")
    #default="3DShapesPyBullet")
  parser.add_argument('--nb_3dshapespybullet_shapes', type=int, default=5)
  parser.add_argument('--nb_3dshapespybullet_colors', type=int, default=8)
  parser.add_argument('--nb_3dshapespybullet_train_colors', type=int, default=6)
  parser.add_argument('--nb_3dshapespybullet_samples', type=int, default=100)
  parser.add_argument("--arch", type=str, 
    choices=["BaselineCNN",
             "SmallBaselineCNN",
             "BN+SmallBaselineCNN",
             "ShortBaselineCNN",
             "BN+BaselineCNN",
             "CNN",
             "CNN3x3",
             "BN+CNN",
             "BN+CNN3x3",
             "BN+3xCNN3x3",
             "BN+BetaVAE3x3",
             "BN+Coord2CNN3x3",
             "BN+Coord4CNN3x3",
             ], 
    help="model architecture to train",
    default="BaselineCNN")
    #default="BN+3xCNN3x3")
  parser.add_argument("--graphtype", type=str,
    choices=["straight_through_gumbel_softmax",
             "reinforce",
             "baseline_reduced_reinforce",
             "normalized_reinforce",
             "baseline_reduced_normalized_reinforce",
             "max_entr_reinforce",
             "baseline_reduced_normalized_max_entr_reinforce",
             "argmax_reinforce",
             "obverter"],
    help="type of graph to use during training of the speaker and listener.",
    default="straight_through_gumbel_softmax")
  parser.add_argument("--max_sentence_length", type=int, default=20)
  parser.add_argument("--vocab_size", type=int, default=5)
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
    #default="CE")
  parser.add_argument("--agent_type", type=str,
    choices=[
      "Baseline",
      "EoSPriored",
      ],
    default="Baseline")
  parser.add_argument("--lr", type=float, default=6e-4)
  parser.add_argument("--gradient_clip", action="store_true", default=False)
  parser.add_argument("--gradient_clip_threshold", type=float, default=1.0)
  parser.add_argument("--epoch", type=int, default=10000)
  parser.add_argument("--metric_epoch_period", type=int, default=20)
  parser.add_argument("--dataloader_num_worker", type=int, default=4)
  parser.add_argument("--nbr_latents", type=int, default=10)
  parser.add_argument("--nbr_values_per_latent", type=int, default=10)
  parser.add_argument("--metric_fast", action="store_true", default=False)
  parser.add_argument("--batch_size", type=int, default=50)
  parser.add_argument("--mini_batch_size", type=int, default=256)
  parser.add_argument("--dropout_prob", type=float, default=0.0)
  parser.add_argument("--emb_dropout_prob", type=float, default=0.0)
  parser.add_argument("--nbr_experience_repetition", type=int, default=1)
  parser.add_argument("--nbr_train_points", type=int, default=10000)
  parser.add_argument("--nbr_eval_points", type=int, default=5000)
  parser.add_argument("--nbr_train_dataset_repetition", type=int, default=1)
  parser.add_argument("--nbr_test_dataset_repetition", type=int, default=1)
  parser.add_argument("--nbr_test_distractors", type=int, default=0)
  parser.add_argument("--nbr_train_distractors", type=int, default=0)
  parser.add_argument("--resizeDim", default=128, type=int,help="input image resize")
  parser.add_argument("--symbol_processing_nbr_hidden_units", default=64, type=int,help="GRU cells")
  parser.add_argument("--symbol_embedding_size", default=64, type=int,help="GRU cells")
  parser.add_argument("--shared_architecture", action="store_true", default=False)
  parser.add_argument("--with_baseline", action="store_true", default=False)
  parser.add_argument("--homoscedastic_multitasks_loss", action="store_true", default=False)
  parser.add_argument("--use_curriculum_nbr_distractors", action="store_true", default=False)
  parser.add_argument("--use_feat_converter", action="store_true", default=False)
  parser.add_argument("--descriptive", action="store_true", default=False)
  parser.add_argument("--descriptive_ratio", type=float, default=0.0)
  parser.add_argument("--object_centric", action="store_true", default=False)
  parser.add_argument("--egocentric", action="store_true", default=False)
  parser.add_argument("--egocentric_tr_degrees", type=int, default=12) #25)
  parser.add_argument("--egocentric_tr_xy", type=float, default=0.0625)
  parser.add_argument("--distractor_sampling", type=str,
    choices=[ "uniform",
              "similarity-0.98",
              "similarity-0.90",
              "similarity-0.75",
              ],
    default="uniform")
  # Obverter Hyperparameters:
  parser.add_argument("--use_sentences_one_hot_vectors", action="store_true", default=False)
  parser.add_argument("--obverter_use_decision_head", action="store_true", default=False)
  parser.add_argument("--differentiable", action="store_true", default=False)
  parser.add_argument("--obverter_threshold_to_stop_message_generation", type=float, default=0.95)
  parser.add_argument("--obverter_nbr_games_per_round", type=int, default=20)
  # Iterade Learning Model:
  parser.add_argument("--iterated_learning_scheme", action="store_true", default=False)
  parser.add_argument("--iterated_learning_period", type=int, default=4)
  parser.add_argument("--iterated_learning_rehearse_MDL", action="store_true", default=False)
  parser.add_argument("--iterated_learning_rehearse_MDL_factor", type=float, default=1.0)
  # Cultural Bottleneck:
  parser.add_argument("--cultural_pressure_it_period", type=int, default=None)
  parser.add_argument("--cultural_reset_meta_learning_rate", type=float, default=1e-3)
  parser.add_argument("--cultural_speaker_substrate_size", type=int, default=1)
  parser.add_argument("--cultural_listener_substrate_size", type=int, default=1)
  parser.add_argument(
    "--cultural_reset_strategy", 
    type=str, 
    default="uniformSL",
    choices=[
      "uniformS",
      "uniformL",  
      "uniformSL",
      "metaS",
      "metaL",  
      "metaSL",
    ],
  ) 
  #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
  
  # Dataset Hyperparameters:
  parser.add_argument("--train_test_split_strategy", type=str, 
    choices=[
      "compositional-10-nb_train_colors_6",
      "combinatorial2-Y-2-8-X-2-8-Orientation-2-10-Scale-1-3-Shape-3-N", #
      "combinatorial2-Y-1-16-X-1-16-Orientation-2-10-Scale-1-3-Shape-3-N",
      "combinatorial2-Y-2-8-X-2-8-Orientation-4-5-Scale-1-3-Shape-3-N",
      "combinatorial2-Y-4-4-X-4-4-Orientation-4-5-Scale-1-3-Shape-3-N", #
      "combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-1-3-Shape-3-N", #
    ],
    help="train/test split strategy",
    # dspites:
    #default="combinatorial2-Y-2-8-X-2-8-Orientation-2-10-Scale-1-3-Shape-3-N")
    #default="combinatorial2-Y-2-8-X-2-8-Orientation-4-5-Scale-1-3-Shape-3-N")
    default="combinatorial2-Y-1-16-X-1-16-Orientation-2-10-Scale-1-3-Shape-3-N")
    #default="combinatorial2-Y-4-4-X-4-4-Orientation-4-5-Scale-1-3-Shape-3-N")
    #default="combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-1-3-Shape-3-N")
    # Test 2 colors:
    #default="compositional-10-nb_train_colors_6")
    # Test 4 colors:
    #default="compositional-10-nb_train_colors_4")
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
  
  parser.add_argument("--vae_nbr_latent_dim", type=int, default=256)
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
      "max_sentence_length":      args.max_sentence_length,
      "nbr_communication_round":  1,
      "nbr_distractors":          {"train":args.nbr_train_distractors, "test":args.nbr_test_distractors},
      "distractor_sampling":      args.distractor_sampling,
      # Default: use "similarity-0.5"
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
      "tau0":                     0.2,
      "gumbel_softmax_eps":       1e-6,
      "vocab_size":               args.vocab_size,
      "force_eos":                args.force_eos,
      "symbol_embedding_size":    args.symbol_embedding_size, #64

      "agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
      "agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
      "agent_loss_type":          args.agent_loss_type, #"NLL"

      "cultural_pressure_it_period": args.cultural_pressure_it_period,
      "cultural_speaker_substrate_size":  args.cultural_speaker_substrate_size,
      "cultural_listener_substrate_size":  args.cultural_listener_substrate_size,
      "cultural_reset_strategy":  args.cultural_reset_strategy, #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
      "cultural_reset_meta_learning_rate":  args.cultural_reset_meta_learning_rate,

      # Cultural Bottleneck:
      "iterated_learning_scheme": args.iterated_learning_scheme,
      "iterated_learning_period": args.iterated_learning_period,
      "iterated_learning_rehearse_MDL": args.iterated_learning_rehearse_MDL,
      "iterated_learning_rehearse_MDL_factor": args.iterated_learning_rehearse_MDL_factor,
      
      # Obverter Hyperparameters:
      "obverter_stop_threshold":  args.obverter_threshold_to_stop_message_generation,  #0.0 if not in use.
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
      
      "with_gradient_clip":       args.gradient_clip,
      "gradient_clip":            args.gradient_clip_threshold,
      
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
  test_split_strategy = args.train_test_split_strategy

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
  
  if "3xCNN" in agent_config["architecture"]:
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
    agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
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
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
    agent_config["symbol_processing_nbr_rnn_layers"] = 1

  elif "3DivBaselineCNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    if "BN" in args.arch:
      agent_config["cnn_encoder_channels"] = ["BN32","BN32","BN32","BN32","BN32","BN32","BN32"]
    else:
      agent_config["cnn_encoder_channels"] = [32,32,32,32,32,32,32]
    
    agent_config["cnn_encoder_kernels"] = [3,3,3,3,3,3,3]
    agent_config["cnn_encoder_strides"] = [2,1,1,2,1,1,2]
    agent_config["cnn_encoder_paddings"] = [1,1,1,1,1,1,1]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = 256 #args.symbol_processing_nbr_hidden_units
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = args.symbol_processing_nbr_hidden_units

    if "MHDPA" in agent_config["architecture"]:
      agent_config["mhdpa_nbr_head"] = 4
      agent_config["mhdpa_nbr_rec_update"] = 1
      agent_config["mhdpa_nbr_mlp_unit"] = 256
      agent_config["mhdpa_interaction_dim"] = 128

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
    agent_config["symbol_processing_nbr_rnn_layers"] = 1
  
  elif "ShortBaselineCNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    agent_config["cnn_encoder_channels"] = ["BN20","BN20","BN20","BN20","BN20"]
    
    agent_config["cnn_encoder_kernels"] = [3,3,3,3,3]
    agent_config["cnn_encoder_strides"] = [2,2,2,2,2]
    agent_config["cnn_encoder_paddings"] = [1,1,1,1,1]
    agent_config["cnn_encoder_non_linearities"] = [torch.nn.ReLU]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = 50 #args.symbol_processing_nbr_hidden_units
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = args.symbol_processing_nbr_hidden_units

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
    agent_config["symbol_processing_nbr_rnn_layers"] = 1

  elif "SmallBaselineCNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    if "BN" in args.arch:
      agent_config["cnn_encoder_channels"] = ["BN32","BN32","BN32","BN32"]
    else:
      agent_config["cnn_encoder_channels"] = [32,32,32,32]
    
    agent_config["cnn_encoder_kernels"] = [8,3,3,3]
    agent_config["cnn_encoder_strides"] = [4,2,2,1]
    agent_config["cnn_encoder_paddings"] = [1,1,1,1]
    agent_config["cnn_encoder_non_linearities"] = [torch.nn.ReLU]
    agent_config["cnn_encoder_fc_hidden_units"] = [128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = 64 #args.symbol_processing_nbr_hidden_units
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = args.symbol_processing_nbr_hidden_units

    if "MHDPA" in agent_config["architecture"]:
      agent_config["mhdpa_nbr_head"] = 4
      agent_config["mhdpa_nbr_rec_update"] = 1
      agent_config["mhdpa_nbr_mlp_unit"] = 256
      agent_config["mhdpa_interaction_dim"] = 128

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
    agent_config["symbol_processing_nbr_rnn_layers"] = 1
  
  elif "BaselineCNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    if "BN" in args.arch:
      agent_config["cnn_encoder_channels"] = ["BN32","BN32","BN32","BN32","BN32","BN32","BN32","BN32"]
    else:
      agent_config["cnn_encoder_channels"] = [32,32,32,32,32,32,32,32]
    
    agent_config["cnn_encoder_kernels"] = [3,3,3,3,3,3,3,3]
    agent_config["cnn_encoder_strides"] = [2,1,1,2,1,2,1,2]
    agent_config["cnn_encoder_paddings"] = [1,1,1,1,1,1,1,1]
    agent_config["cnn_encoder_non_linearities"] = [torch.nn.ReLU]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = 256 #args.symbol_processing_nbr_hidden_units
    # N.B.: if cnn_encoder_fc_hidden_units is [],
    # then this last parameter does not matter.
    # The cnn encoder is not topped by a FC network.

    agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["feat_converter_output_size"] = args.symbol_processing_nbr_hidden_units

    if "MHDPA" in agent_config["architecture"]:
      agent_config["mhdpa_nbr_head"] = 4
      agent_config["mhdpa_nbr_rec_update"] = 1
      agent_config["mhdpa_nbr_mlp_unit"] = 256
      agent_config["mhdpa_interaction_dim"] = 128

    agent_config["temporal_encoder_nbr_hidden_units"] = 0
    agent_config["temporal_encoder_nbr_rnn_layers"] = 0
    agent_config["temporal_encoder_mini_batch_size"] = args.mini_batch_size
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
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
    agent_config["cnn_encoder_non_linearities"] = [torch.nn.ReLU]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
    agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
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
    agent_config["symbol_processing_nbr_hidden_units"] = args.symbol_processing_nbr_hidden_units
    agent_config["symbol_processing_nbr_rnn_layers"] = 1
    
  else:
    raise NotImplementedError

  save_path_dataset = ''
  if '3DShapesPyBullet' in args.dataset:
    generate = False
    img_size = 128 #64
    nb_shapes = args.nb_3dshapespybullet_shapes
    nb_colors = args.nb_3dshapespybullet_colors
    nb_samples = args.nb_3dshapespybullet_samples
    nb_train_colors = args.nb_3dshapespybullet_train_colors
    train_split_strategy = f'compositional-40-nb_train_colors_{nb_train_colors}' 
    test_split_strategy = train_split_strategy

    root = './datasets/3DShapePyBullet-dataset'
    root += f'imgS{img_size}-shapes{nb_shapes}-colors{nb_colors}-samples{nb_samples}'
    save_path_dataset = f'3DShapePyBullet-dataset-imgS{img_size}-shapes{nb_shapes}-colors{nb_colors}-samples{nb_samples}'
    

  save_path = ""
  if args.parent_folder != '':
    save_path += args.parent_folder+'/'
  save_path += f"{args.dataset}+DualLabeled/"
  #save_path += f"/MetricEPS1m5/TestModularityDisentanglementMetricShuffleOHE_nbrTrainPoints{args.nbr_train_points}+PERM+SQRT"
  save_path += f"/MetricEPS1m20/TestPosBosDisentanglementMetricShuffleOHE_nbrTrainPoints{args.nbr_train_points}+PERM+Sq+RT4+Minmi/"
  
  save_path += f"NOTDroppingEoS/BosdisShuffleDistinctTest"
  save_path += "+TestVSMIGForm+MIGSameTestExpectPos+ActualModularitySameTestExpectNeg+ModForm2"
  
  if args.use_obverter_sampling:
    save_path += "WithObverterSampling/"

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
  elif 'Dummy' in args.dataset: 
    train_test_strategy = f"-{test_split_strategy}"
    if test_split_strategy != train_split_strategy:
      train_test_strategy = f"/train_{train_split_strategy}/test_{test_split_strategy}"
    save_path += f"/DummyDataset_L{args.nbr_latents}X{args.nbr_values_per_latent}"#{train_test_strategy}"
  elif '3DShapesPyBullet' in args.dataset: 
    train_test_strategy = f"-{train_split_strategy}"
    save_path += save_path_dataset
  
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

  if args.gradient_clip:
    save_path += f"_gradClip{args.gradient_clip_threshold}"

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
    save_path += f"Obverter{'WithDecisionHead' if args.obverter_use_decision_head else 'WithBMM'}{args.obverter_threshold_to_stop_message_generation}-{args.obverter_nbr_games_per_round}GPR/DEBUG_{'OHE' if args.use_sentences_one_hot_vectors else ''}/"
  else:
    save_path += f"STGS-{args.agent_type}-LSTM-CNN-Agent/"

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

  """
  #from ReferentialGym.agents import DifferentiableObverterAgent
  from ReferentialGym.agents.halfnew_differentiable_obverter_agent import DifferentiableObverterAgent
  #from ReferentialGym.agents.depr_differentiable_obverter_agent import DifferentiableObverterAgent
  
  
  if 'obverter' in args.graphtype:
    speaker = DifferentiableObverterAgent(
      kwargs=agent_config, 
      obs_shape=obs_shape, 
      vocab_size=vocab_size, 
      max_sentence_length=max_sentence_length,
      agent_id='s0',
      logger=logger,
      use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
      use_decision_head_=args.obverter_use_decision_head,
      differentiable=args.differentiable
    )
  elif 'Baseline' in args.agent_type:
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
      use_decision_head_=args.obverter_use_decision_head,
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
  """

  # # Dataset:
  need_dict_wrapping = {}

  if 'dSprites' in args.dataset:
    root = './datasets/dsprites-dataset'
    train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
    test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)
  elif 'Dummy' in args.dataset:
    train_dataset = DummyDataset(
        train=True, 
        transform=rg_config['train_transform'], 
        split_strategy=None, 
        nbr_latents=args.nbr_latents,
        nbr_values_per_latent=args.nbr_values_per_latent,
    )
    test_dataset = DummyDataset(
        train=False, 
        transform=rg_config['test_transform'], 
        split_strategy=None,
        nbr_latents=args.nbr_latents,
        nbr_values_per_latent=args.nbr_values_per_latent,
    )
  elif '3DShapesPyBullet' in args.dataset:
    train_dataset = ReferentialGym.datasets._3DShapesPyBulletDataset(
      root=root, 
      train=True, 
      transform=rg_config['train_transform'],
      generate=generate,
      img_size=img_size,
      nb_samples=nb_samples,
      nb_shapes=nb_shapes,
      nb_colors=nb_colors,
      split_strategy=train_split_strategy,
    )
    
    test_dataset = ReferentialGym.datasets._3DShapesPyBulletDataset(
      root=root, 
      train=False, 
      transform=rg_config['test_transform'],
      generate=False,
      img_size=img_size,
      nb_samples=nb_samples,
      nb_shapes=nb_shapes,
      nb_colors=nb_colors,
      split_strategy=test_split_strategy,
    )
  else:
    raise NotImplementedError 
  
  ## Modules:
  modules = {}

  from ReferentialGym import modules as rg_modules

  # NoiseSource:
  pnsm_id = "progressive_noise_source_0"
  pnsm_config = {
    "shuffle_period_increment": 0.01,
  }
  pnsm_input_stream_ids = {
    "input":"current_dataloader:sample:speaker_exp_latents", 
  }

  psm_id = "progressive_shuffle_0"
  psm_config = {
    "shuffle_period_increment": 0.01,
  }
  psm_input_stream_ids = {
    "input":"current_dataloader:sample:speaker_exp_latents", 
  }
  
  
  """
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

  """

  modules[pnsm_id] = ProgressiveBatchShuffleModule(
    id=pnsm_id,
    config=pnsm_config,
    input_stream_ids=pnsm_input_stream_ids
  )

  modules[psm_id] = ProgressiveShuffleModule(
    id=psm_id,
    config=psm_config,
    input_stream_ids=psm_input_stream_ids
  )

  
  """
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

  """

  ## Pipelines:
  pipelines = {}

  """
  # 0) Now that all the modules are known, let us build the optimization module:
  optim_id = "global_optim"
  optim_config = {
    "modules":modules,
    "learning_rate":args.lr,
    "optimizer_type":args.optimizer_type,
    "with_gradient_clip":rg_config["with_gradient_clip"],
    "gradient_clip":rg_config["gradient_clip"],
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
  """
  
  speaker_posbosdis_disentanglement_metric_id = "speaker_posbosdis_disentanglement_metric"
  speaker_posbosdis_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  def preprocess_fn(x):
    vocab_size = 10
    ndim = x.shape[-1]
    for didx in range(ndim):
        x[:, ..., didx] = ndim*didx + x[:, ..., didx]
    return x

  speaker_posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
    id=speaker_posbosdis_disentanglement_metric_id,
    input_stream_ids=speaker_posbosdis_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": preprocess_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_posbosdis_disentanglement_metric_id] = speaker_posbosdis_disentanglement_metric_module

  #///

  speaker_toposim_metric_id = "speaker_toposim_metric"
  speaker_toposim_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    "features":f"modules:{pnsm_id}:output",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
    "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  speaker_toposim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
    id=speaker_toposim_metric_id,
    input_stream_ids=speaker_toposim_metric_input_stream_ids,
    config = {
        "pvalue_significance_threshold": 0.5,
        "parallel_TS_computation_max_workers":16,
        #"filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        "postprocess_fn": preprocess_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
        "preprocess_fn": (lambda x: x),
        # not necessary if providing a preprocess_fn, 
        # that computes the features/_sense output, but here it is in order to deal with shapes:
        #"features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
        "epoch_period":args.metric_epoch_period,
        "batch_size":64,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        #"resample":False,
        "resample":True,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":True,
    }
  )
  modules[speaker_toposim_metric_id] = speaker_toposim_metric_module

  #///////////////////////////////////////////////////


  # Testing for the case where all symbols encode for the same attribute
  # (or in this case for only 2/10 attributes in a repeated fashion) :
  # Should see a difference between MIG formulation (summing over attribute gaps)
  # and PosBosDis formulation (summing over representation's dim gaps):
  speaker_same_posbosdis_disentanglement_metric_id = "speaker_same_posbosdis_disentanglement_metric"
  speaker_same_posbosdis_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  def preprocess_same_fn(x):
    ndim = x.shape[-1]
    for didx in range(2,ndim):
        if didx % 2:
            x[:, ..., didx] = x[:, ..., 0]
        else:
            x[:, ..., didx] = x[:, ..., 1]
    return x

  speaker_same_posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
    id=speaker_same_posbosdis_disentanglement_metric_id,
    input_stream_ids=speaker_same_posbosdis_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": preprocess_same_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_same_posbosdis_disentanglement_metric_id] = speaker_same_posbosdis_disentanglement_metric_module

  #/////

  speaker_same_toposim_metric_id = "speaker_same_toposim_metric"
  speaker_same_toposim_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    "features":f"modules:{pnsm_id}:output",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
    "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  speaker_same_toposim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
    id=speaker_same_toposim_metric_id,
    input_stream_ids=speaker_same_toposim_metric_input_stream_ids,
    config = {
        "pvalue_significance_threshold": 0.5,
        "parallel_TS_computation_max_workers":16,
        #"filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        "postprocess_fn": preprocess_same_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
        "preprocess_fn": (lambda x: x),
        # not necessary if providing a preprocess_fn, 
        # that computes the features/_sense output, but here it is in order to deal with shapes:
        #"features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
        "epoch_period":args.metric_epoch_period,
        "batch_size":64,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        #"resample":False,
        "resample":True,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":True,
    }
  )
  modules[speaker_same_toposim_metric_id] = speaker_same_toposim_metric_module

  #///////////////////////////////////////////////////



  # shuffle :
  speaker_shuffle_posbosdis_disentanglement_metric_id = "speaker_shuffle_posbosdis_disentanglement_metric"
  speaker_shuffle_posbosdis_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    #"latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_shuffle_posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
    id=speaker_shuffle_posbosdis_disentanglement_metric_id,
    input_stream_ids=speaker_shuffle_posbosdis_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": (lambda x: x), #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_shuffle_posbosdis_disentanglement_metric_id] = speaker_shuffle_posbosdis_disentanglement_metric_module


  #///

  speaker_shuffle_toposim_metric_id = "speaker_shuffle_toposim_metric"
  speaker_shuffle_toposim_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    "features":f"modules:{psm_id}:output",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
    "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  speaker_shuffle_toposim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
    id=speaker_shuffle_toposim_metric_id,
    input_stream_ids=speaker_shuffle_toposim_metric_input_stream_ids,
    config = {
        "pvalue_significance_threshold": 0.5,
        "parallel_TS_computation_max_workers":16,
        #"filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        "postprocess_fn": (lambda x: x),  #(lambda x: x["sentences_widx"].cpu().numpy())
        "preprocess_fn": (lambda x: x),
        # not necessary if providing a preprocess_fn, 
        # that computes the features/_sense output, but here it is in order to deal with shapes:
        #"features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
        "epoch_period":args.metric_epoch_period,
        "batch_size":64,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        #"resample":False,
        "resample":True,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":True,
    }
  )
  modules[speaker_shuffle_toposim_metric_id] = speaker_shuffle_toposim_metric_module

  #///////////////////////////////////////////////////

  # testing Bosdis being unaffected by shuffling symbols position 
  # when "symbols univocally refer to distinct input elements" [Chaabouni et al. 2020]:
  # shuffle :
  speaker_distinct_shuffle_posbosdis_disentanglement_metric_id = "speaker_distinct_shuffle_posbosdis_disentanglement_metric"
  speaker_distinct_shuffle_posbosdis_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    #"latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_distinct_shuffle_posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
    id=speaker_distinct_shuffle_posbosdis_disentanglement_metric_id,
    input_stream_ids=speaker_distinct_shuffle_posbosdis_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": (lambda x: x), #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": preprocess_fn, #(lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_distinct_shuffle_posbosdis_disentanglement_metric_id] = speaker_distinct_shuffle_posbosdis_disentanglement_metric_module

  #///

  speaker_distinct_shuffle_toposim_metric_id = "speaker_distinct_shuffle_toposim_metric"
  speaker_distinct_shuffle_toposim_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    "features":f"modules:{psm_id}:output",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
    "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  speaker_distinct_shuffle_toposim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
    id=speaker_distinct_shuffle_toposim_metric_id,
    input_stream_ids=speaker_distinct_shuffle_toposim_metric_input_stream_ids,
    config = {
        "pvalue_significance_threshold": 0.5,
        "parallel_TS_computation_max_workers":16,
        #"filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        "postprocess_fn": (lambda x: x),  #(lambda x: x["sentences_widx"].cpu().numpy())
        "preprocess_fn": preprocess_fn,
        # not necessary if providing a preprocess_fn, 
        # that computes the features/_sense output, but here it is in order to deal with shapes:
        #"features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
        "epoch_period":args.metric_epoch_period,
        "batch_size":64,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        #"resample":False,
        "resample":True,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":True,
    }
  )
  modules[speaker_distinct_shuffle_toposim_metric_id] = speaker_distinct_shuffle_toposim_metric_module

  #///////////////////////////////////////////////////

  # Modularity:
  speaker_modularity_disentanglement_metric_id = "speaker_modularity_disentanglement_metric"
  speaker_modularity_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
    id=speaker_modularity_disentanglement_metric_id,
    input_stream_ids=speaker_modularity_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": (lambda x: x),
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_modularity_disentanglement_metric_id] = speaker_modularity_disentanglement_metric_module

  # Testing for the case where all symbols encode for the same attribute
  # (or in this case for only 2/10 attributes in a repeated fashion) :
  # Should see a difference between MIG formulation (summing over attribute gaps)
  # and PosBosDis formulation (summing over representation's dim gaps). 
  # In the below, we expect the Modularity metric to NOT differentiate them:
  speaker_same_modularity_disentanglement_metric_id = "speaker_same_modularity_disentanglement_metric"
  speaker_same_modularity_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  def preprocess_same_fn(x):
    ndim = x.shape[-1]
    for didx in range(2,ndim):
        if didx % 2:
            x[:, ..., didx] = x[:, ..., 0]
        else:
            x[:, ..., didx] = x[:, ..., 1]
    return x

  speaker_same_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
    id=speaker_same_modularity_disentanglement_metric_id,
    input_stream_ids=speaker_same_modularity_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": preprocess_same_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_same_modularity_disentanglement_metric_id] = speaker_same_modularity_disentanglement_metric_module
  
  # shuffle :
  speaker_shuffle_modularity_disentanglement_metric_id = "speaker_shuffle_modularity_disentanglement_metric"
  speaker_shuffle_modularity_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    #"latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_shuffle_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
    id=speaker_shuffle_modularity_disentanglement_metric_id,
    input_stream_ids=speaker_shuffle_modularity_disentanglement_metric_input_stream_ids,
    config = {
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_shuffle_modularity_disentanglement_metric_id] = speaker_shuffle_modularity_disentanglement_metric_module





  # Mutual Information Gap:
  speaker_mig_disentanglement_metric_id = "speaker_mig_disentanglement_metric"
  speaker_mig_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
    id=speaker_mig_disentanglement_metric_id,
    input_stream_ids=speaker_mig_disentanglement_metric_input_stream_ids,
    config = {
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      #"nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_mig_disentanglement_metric_id] = speaker_mig_disentanglement_metric_module

  # Testing for the case where all symbols encode for the same attribute
  # (or in this case for only 2/10 attributes in a repeated fashion) :
  # Should see a difference between MIG formulation (summing over attribute gaps)
  # and PosBosDis formulation (summing over representation's dim gaps). In the below, we expect the MIG metric
  # to differentiate too:
  speaker_same_mig_disentanglement_metric_id = "speaker_same_mig_disentanglement_metric"
  speaker_same_mig_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{pnsm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{pnsm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
    "indices":"current_dataloader:sample:speaker_indices", 
  }

  def preprocess_same_fn(x):
    ndim = x.shape[-1]
    for didx in range(2,ndim):
        if didx % 2:
            x[:, ..., didx] = x[:, ..., 0]
        else:
            x[:, ..., didx] = x[:, ..., 1]
    return x

  speaker_same_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
    id=speaker_same_mig_disentanglement_metric_id,
    input_stream_ids=speaker_same_mig_disentanglement_metric_input_stream_ids,
    config = {
      "postprocess_fn": preprocess_same_fn,  #(lambda x: x["sentences_widx"].cpu().numpy())
      "preprocess_fn": (lambda x: x),
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      "nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_same_mig_disentanglement_metric_id] = speaker_same_mig_disentanglement_metric_module
  
  # shuffle :
  speaker_shuffle_mig_disentanglement_metric_id = "speaker_shuffle_mig_disentanglement_metric"
  speaker_shuffle_mig_disentanglement_metric_input_stream_ids = {
    #"model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
    "model":f"modules:{psm_id}:ref",
    #"representations":"modules:current_speaker:ref:ref_agent:features",
    "representations":f"modules:{psm_id}:output",
    "experiences":"current_dataloader:sample:speaker_experiences", 
    #"latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
    "indices":"current_dataloader:sample:speaker_indices", 
  }
  speaker_shuffle_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
    id=speaker_shuffle_mig_disentanglement_metric_id,
    input_stream_ids=speaker_shuffle_mig_disentanglement_metric_input_stream_ids,
    config = {
      "epoch_period":args.metric_epoch_period,
      "batch_size":64,#5,
      "nbr_train_points":args.nbr_train_points,#3000,
      #"nbr_eval_points":args.nbr_eval_points,#2000,
      #"resample":False,
      "resample":True,
      "threshold":5e-2,#0.0,#1.0,
      "random_state_seed":args.seed,
      "verbose":False,
      "active_factors_only":True,
    }
  )
  modules[speaker_shuffle_mig_disentanglement_metric_id] = speaker_shuffle_mig_disentanglement_metric_module

  
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

  pipelines["referential_game"] = []

  """
  pipelines["referential_game"] += [
    population_handler_id,
    current_speaker_id,
    current_listener_id
  ]
  """

  """
  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)
  """
  pipelines[speaker_posbosdis_disentanglement_metric_id] = []
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id] = []
  
  pipelines[speaker_modularity_disentanglement_metric_id] = []
  pipelines[speaker_shuffle_modularity_disentanglement_metric_id] = []
  
  """
  # Add gradient recorder module for debugging purposes:
  pipelines[optim_id].append(grad_recorder_id)
  """
  pipelines[speaker_posbosdis_disentanglement_metric_id].append(pnsm_id)
  pipelines[speaker_posbosdis_disentanglement_metric_id].append(speaker_posbosdis_disentanglement_metric_id)
  pipelines[speaker_posbosdis_disentanglement_metric_id].append(speaker_same_posbosdis_disentanglement_metric_id)
  
  pipelines[speaker_posbosdis_disentanglement_metric_id].append(speaker_toposim_metric_id)
  pipelines[speaker_posbosdis_disentanglement_metric_id].append(speaker_same_toposim_metric_id)
  
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id].append(psm_id)
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id].append(speaker_shuffle_posbosdis_disentanglement_metric_id)
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id].append(speaker_distinct_shuffle_posbosdis_disentanglement_metric_id)
  
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id].append(speaker_shuffle_toposim_metric_id)
  pipelines[speaker_shuffle_posbosdis_disentanglement_metric_id].append(speaker_distinct_shuffle_toposim_metric_id)
  


  pipelines[speaker_modularity_disentanglement_metric_id].append(pnsm_id)
  pipelines[speaker_modularity_disentanglement_metric_id].append(speaker_modularity_disentanglement_metric_id)
  pipelines[speaker_modularity_disentanglement_metric_id].append(speaker_same_modularity_disentanglement_metric_id)
  
  # MIG:
  pipelines[speaker_modularity_disentanglement_metric_id].append(speaker_mig_disentanglement_metric_id)
  pipelines[speaker_modularity_disentanglement_metric_id].append(speaker_same_mig_disentanglement_metric_id)
  
  pipelines[speaker_shuffle_modularity_disentanglement_metric_id].append(psm_id)
  pipelines[speaker_shuffle_modularity_disentanglement_metric_id].append(speaker_shuffle_modularity_disentanglement_metric_id)
  # MIG:
  pipelines[speaker_shuffle_modularity_disentanglement_metric_id].append(speaker_shuffle_mig_disentanglement_metric_id)
  
  
  #pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
  #pipelines[optim_id].append(topo_sim_metric_id)
  #pipelines[optim_id].append(inst_coord_metric_id)
  #if 'dSprites' in args.dataset:  pipelines[optim_id].append(dsprites_latent_metric_id)
  
  pipelines[logger_id] = [logger_id]

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
