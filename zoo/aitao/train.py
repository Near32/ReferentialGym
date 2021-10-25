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


###########################################################
###########################################################
###########################################################
"""
HOW-TO:
- train on 3DShapes:
  - poorly diverse stimuli:
    python -m ipdb -c c RG/zoo/referential-games+compositionality+disentanglement/train.py --parent_folder ./PoorlyDiverseStimuli \
    --use_cuda --seed 20 --obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.75 --batch_size 64 --mini_batch_size 64 --resizeDim 64 --arch BetaVAEEncoderOnly3x3 --max_sentence_length 10 --vocab_size 10 --epoch 4001 \
    --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 --nbr_train_distractors 0 --nbr_test_distractors 0 --obverter_use_decision_head --obverter_nbr_head_outputs 2 --agent_loss_type NLL --graphtype straight_through_gumbel_softmax --metric_epoch_period 50 \
    --nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 --dataset 3DShapesPyBullet \
    --lr 6e-4 --nbr_train_points 1000 --nbr_eval_points 500 --metric_batch_size 16 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 --with_baseline --baseline_only --vae_factor_gamma 60.0 --descriptive --descriptive_ratio 0.5 --dis_metric_resampling --metric_resampling --object_centric --shared_architecture --metric_active_factors_only

  - richly diverse stimuli:
    python -m ipdb -c c RG/zoo/referential-games+compositionality+disentanglement/train.py --parent_folder ./PoorlyDiverseStimuli \
    --use_cuda --seed 20 --obverter_nbr_games_per_round 5 --obverter_threshold_to_stop_message_generation 0.75 --batch_size 64 --mini_batch_size 64 --resizeDim 64 --arch BetaVAEEncoderOnly3x3 --max_sentence_length 10 --vocab_size 10 --epoch 4001 \
    --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 --nbr_train_distractors 0 --nbr_test_distractors 0 --obverter_use_decision_head --obverter_nbr_head_outputs 2 --agent_loss_type NLL --graphtype straight_through_gumbel_softmax --metric_epoch_period 50 \
    --nb_3dshapespybullet_shapes 10 --nb_3dshapespybullet_colors 10 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 5 --dataset 3DShapesPyBullet \
    --lr 6e-4 --nbr_train_points 4000 --nbr_eval_points 2000 --metric_batch_size 16 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 --with_baseline --baseline_only --vae_factor_gamma 60.0 --descriptive --descriptive_ratio 0.5 --dis_metric_resampling --metric_resampling --object_centric --shared_architecture --metric_active_factors_only

"""
###########################################################
###########################################################
###########################################################

from ReferentialGym.modules import Module 

class InterventionModule(Module):
  def __init__(self, id:str, config:Dict[str,object], input_stream_ids:Dict[str,str]):
    default_input_stream_ids = {
      "logs_dict":"logs_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",
      "input":"current_dataloader:sample:speaker_exp_latents_values", 
      "sentences_one_hot":"modules:current_speaker:sentences_one_hot",
      "sentences_widx":"modules:current_speaker:sentences_widx",
    }
    if input_stream_ids is None:
      input_stream_ids = default_input_stream_ids
    else:
      for default_stream, default_id in default_input_stream_ids.items():
        if default_id not in input_stream_ids.values():
          input_stream_ids[default_stream] = default_id

    super(InterventionModule, self).__init__(
      id=id,
      type="InterventionModule",
      config=config,
      input_stream_ids=input_stream_ids
    )

    self.intervention_percentage = 0.01
    if self.config["epoch_progression_end"] < 1.0:
      self.config["epoch_progression_end"] = 1.0
    self.intervention_period_increment = 1.0/self.config["epoch_progression_end"] # expect float ]0,1.0]
    
  def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object]:
    outputs_stream_dict = {}

    logs_dict = input_streams_dict["logs_dict"]
    epoch_idx = input_streams_dict["epoch"]
    mode = input_streams_dict["mode"]
    
    inp = input_streams_dict["input"].long().squeeze(1).squeeze(1)
    sohe = input_streams_dict["sentences_one_hot"]
    swidx = input_streams_dict["sentences_widx"]

    batch_size = inp.shape[0]

    #output_ohe = torch.zeros((batch_size, self.config["max_sentence_length"], self.config["vocab_size"])).to(inp.device)
    output_ohe = sohe
    # (batch_size, max_sentence_length, vocab_size)
    
    #output_widx = torch.zeros((batch_size, self.config["max_sentence_length"], 1)).to(inp.device)
    output_widx = swidx
    # (batch_size, max_sentence_length, 1)
    

    self.intervention_percentage = min(1.0, self.intervention_period_increment*epoch_idx)
    size = int(self.intervention_percentage*inp.shape[0])
    if size != 0:
      for bidx in range(size):
        if output_ohe is not None:  
          # erase:
          output_ohe[bidx] *= 0
        for attr in range(inp.shape[1]):
          if output_ohe is not None:  
            output_ohe[bidx, attr, inp[bidx, attr]] = 1
          #if output_widx is not None:  output_widx[bidx, attr, 0] = inp[bidx, attr]
          if output_widx is not None:  output_widx[bidx, attr] = inp[bidx, attr]
        
        for nsidx in range(attr+1, self.config["max_sentence_length"]):
          if self.config["vocab_stop_idx"] < self.config["max_sentence_length"]: 
            if output_ohe is not None:  output_ohe[bidx, nsidx, self.config["vocab_stop_idx"]] = 1
          else:
            if output_ohe is not None:  output_ohe[bidx, nsidx, 0] = 1
          #if output_widx is not None:  output_widx[bidx, nsidx, 0] = self.config["vocab_stop_idx"]
          if output_widx is not None:  output_widx[bidx, nsidx] = self.config["vocab_stop_idx"]
      
    outputs_stream_dict[self.config["output_widx_placeholder"]] = output_widx
    outputs_stream_dict[self.config["output_ohe_placeholder"]] = output_ohe

    logs_dict[f"{mode}/{self.id}/InterventionPercentage"] = self.intervention_percentage*100.0
    logs_dict[f"{mode}/{self.id}/InterventionPercentage/size"] = size

    return outputs_stream_dict

def make_VAE(agent_config, args, rg_config):
  agent_config["decoder_architecture"] = "BN+DCNN"
  
  agent_config['VAE_lambda'] = args.vae_lambda
  agent_config['vae_beta'] = args.vae_beta
  agent_config['factor_vae_gamma'] = args.vae_factor_gamma
  agent_config['vae_constrainedEncoding'] =  args.vae_constrained_encoding
  agent_config['vae_use_gaussian_observation_model'] = args.vae_gaussian 
  agent_config['vae_observation_sigma'] = args.vae_gaussian_sigma
  agent_config['vae_max_capacity'] = args.vae_max_capacity #1e2
  agent_config['vae_nbr_epoch_till_max_capacity'] = args.vae_nbr_epoch_till_max_capacity

  agent_config['vae_decoder_conv_dim'] = args.vae_decoder_conv_dim
  agent_config['vae_decoder_nbr_layer'] = args.vae_decoder_nbr_layer
  agent_config['vae_nbr_latent_dim'] = args.vae_nbr_latent_dim
  agent_config['vae_detached_featout'] = args.vae_detached_featout
  agent_config['vae_use_mu_value'] = args.vae_use_mu_value

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
  agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
  #agent_config["cnn_encoder_feature_dim"] = cnn_feature_size
  # N.B.: if cnn_encoder_fc_hidden_units is [],
  # then this last parameter does not matter.
  # The cnn encoder is not topped by a FC network.

  agent_config["cnn_encoder_mini_batch_size"] = args.mini_batch_size
  #agent_config["feat_converter_output_size"] = cnn_feature_size
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

  ## Decoder:
  ### CNN:
  if "BN" in agent_config["decoder_architecture"]:
    agent_config["cnn_decoder_channels"] = ["BN64","BN64","BN32","BN32"]
  else:
    agent_config["cnn_decoder_channels"] = [64,64,32,32]
  
  if "3x3" in agent_config["decoder_architecture"]:
    agent_config["cnn_decoder_kernels"] = [3,3,3,3]
  elif "3x4x4x7" in agent_config["decoder_architecture"]:
    agent_config["cnn_decoder_kernels"] = [3,4,4,7]
  else:
    agent_config["cnn_decoder_kernels"] = [4,4,4,4]
  agent_config["cnn_decoder_strides"] = [2,2,2,2]
  agent_config["cnn_decoder_paddings"] = [1,1,1,1]
  
  ### MLP:
  if "BN" in agent_config["decoder_architecture"]:
    agent_config['mlp_decoder_fc_hidden_units'] = ["BN256", "BN256"]
  else:
    agent_config['mlp_decoder_fc_hidden_units'] = [256, 256]
  agent_config['mlp_decoder_fc_hidden_units'].append(40*6)

  return agent_config 


def main():
  parser = argparse.ArgumentParser(description="Language Emergence, Compositionality and Disentanglement.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--parent_folder", type=str, help="folder to save into.",default="TestObverter")
  parser.add_argument("--verbose", action="store_true", default=False)
  parser.add_argument('--synthetic_progression_end', type=int, default=1)
  parser.add_argument("--use_priority", action="store_true", default=False)
  parser.add_argument("--restore", action="store_true", default=False)
  parser.add_argument("--force_eos", action="store_true", default=False)
  parser.add_argument("--use_cuda", action="store_true", default=False)
  parser.add_argument("--dataset", type=str, 
    choices=["Sort-of-CLEVR",
             "tiny-Sort-of-CLEVR",
             "XSort-of-CLEVR",
             "tiny-XSort-of-CLEVR",
             "dSprites",
             "3DShapesPyBullet",
             ], 
    help="dataset to train on.",
    default="3DShapesPyBullet")
  parser.add_argument('--nb_3dshapespybullet_shapes', type=int, default=5)
  parser.add_argument('--nb_3dshapespybullet_colors', type=int, default=8)
  parser.add_argument('--nb_3dshapespybullet_train_colors', type=int, default=6)
  parser.add_argument('--nb_3dshapespybullet_samples', type=int, default=100)
  parser.add_argument("--arch", type=str, 
    choices=["BaselineCNN",
             "ShortBaselineCNN",
             "BN+BaselineCNN",
             "BN+Baseline1CNN",
             "CNN",
             "CNN3x3",
             "BN+CNN",
             "BN+CNN3x3",
             "BN+3xCNN3x3",
             "BN+BetaVAE3x3",
             "BetaVAEEncoderOnly3x3",
             "BN+BetaVAEEncoderOnly3x3",
             "BN+Coord2CNN3x3",
             "BN+Coord4CNN3x3",
             ], 
    help="model architecture to train",
    #default="BaselineCNN")
    #default="BN+3xCNN3x3")
    default="BN+BetaVAE3x3")
  parser.add_argument("--graphtype", type=str,
    choices=["straight_through_gumbel_softmax",
             "reinforce",
             "baseline_reduced_reinforce",
             "normalized_reinforce",
             "baseline_reduced_normalized_reinforce",
             "max_entr_reinforce",
             "baseline_reduced_normalized_max_entr_reinforce",
             "argmax_reinforce",
             "obverter",
             "synthetic_obverter",
             "synthetic_straight_through_gumbel_softmax"],
    help="type of graph to use during training of the speaker and listener.",
    default="obverter")
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
  parser.add_argument("--epoch", type=int, default=10000)
  parser.add_argument("--dataloader_num_worker", type=int, default=8)
  #parser.add_argument("--dataloader_num_worker", type=int, default=1)
 
  parser.add_argument("--metric_epoch_period", type=int, default=20)
  parser.add_argument("--nbr_train_points", type=int, default=3000)
  parser.add_argument("--nbr_eval_points", type=int, default=1000)
  parser.add_argument("--metric_resampling", action="store_true", default=False)
  parser.add_argument("--metric_active_factors_only", action="store_true", default=False)
  parser.add_argument("--dis_metric_resampling", action="store_true", default=False)
  parser.add_argument("--metric_fast", action="store_true", default=False)
  parser.add_argument("--metric_batch_size", type=int, default=8)
  parser.add_argument("--parallel_TS_worker", type=int, default=16)

  parser.add_argument("--batch_size", type=int, default=50)
  parser.add_argument("--mini_batch_size", type=int, default=256)
  parser.add_argument("--dropout_prob", type=float, default=0.0)
  parser.add_argument("--emb_dropout_prob", type=float, default=0.0)
  parser.add_argument("--nbr_experience_repetition", type=int, default=1)
  parser.add_argument("--nbr_train_dataset_repetition", type=int, default=1)
  parser.add_argument("--nbr_test_dataset_repetition", type=int, default=1)
  
  parser.add_argument("--add_descriptive_test", action="store_true", default=False)
  parser.add_argument("--add_discriminative_test", action="store_true", default=False)
  parser.add_argument("--add_descr_discriminative_test", action="store_true", default=False)    

  parser.add_argument("--nbr_discriminative_test_distractors", type=int, default=7)
  parser.add_argument("--nbr_descr_discriminative_test_distractors", type=int, default=7)
  parser.add_argument("--nbr_test_distractors", type=int, default=0)
  parser.add_argument("--nbr_train_distractors", type=int, default=0)
  
  parser.add_argument("--resizeDim", default=128, type=int,help="input image resize")
  parser.add_argument("--agent_nbr_latent_dim", type=int, default=50)
  parser.add_argument("--symbol_processing_nbr_hidden_units", default=64, type=int,help="GRU cells")
  parser.add_argument("--symbol_embedding_size", default=64, type=int,help="GRU cells")
  parser.add_argument("--shared_architecture", action="store_true", default=False)
  
  parser.add_argument("--use_symbolic_stimuli", action="store_true", default=False)
  parser.add_argument("--with_baseline", action="store_true", default=False)
  parser.add_argument("--baseline_only", action="store_true", default=False)
  
  parser.add_argument("--homoscedastic_multitasks_loss", action="store_true", default=False)
  parser.add_argument("--use_curriculum_nbr_distractors", action="store_true", default=False)
  parser.add_argument("--use_feat_converter", action="store_true", default=False)
  parser.add_argument("--descriptive", action="store_true", default=False)
  parser.add_argument("--with_descriptive_not_target_logit_language_conditioning", action="store_true", default=False)
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
              "similarity-0.50",
              ],
    default="similarity-0.50")
  # Obverter Hyperparameters:
  parser.add_argument("--use_sentences_one_hot_vectors", action="store_true", default=False)
  parser.add_argument("--obverter_use_decision_head", action="store_true", default=False)
  parser.add_argument("--obverter_nbr_head_outputs", type=int, default=2)
  
  parser.add_argument("--differentiable", action="store_true", default=False)
  parser.add_argument("--context_consistent_obverter", action="store_true", default=False)
  parser.add_argument("--visual_context_consistent_obverter", action="store_true", default=False)
  parser.add_argument("--normalised_context_consistent_obverter", action="store_true", default=False)
  parser.add_argument("--with_BN_in_obverter_decision_head", action="store_true", default=False)
  parser.add_argument("--with_DP_in_obverter_decision_head", action="store_true", default=False)
  parser.add_argument("--with_DP_in_obverter_decision_head_listener_only", action="store_true", default=False)

  parser.add_argument("--obverter_threshold_to_stop_message_generation", type=float, default=0.95)
  parser.add_argument("--obverter_nbr_games_per_round", type=int, default=20)
  parser.add_argument("--use_obverter_sampling", action="store_true", default=False)
  parser.add_argument("--use_aitao", action="store_true", default=False)
  parser.add_argument("--obverter_sampling_round_alternation_only", action="store_true", default=False)
  # Iterade Learning Model:
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
  # No longer used for 3d shapes but only for dSprites: default means 80-20% 
  parser.add_argument("--train_test_split_strategy", type=str, 
    choices=[
      "combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N",
      "combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N",
      "combinatorial2-Y-1-S16-X-1-S16-Orientation-4-N-Scale-1-S3-Shape-1-S1", 
      "combinatorial2-Y-8-S2-X-8-S2-Orientation-4-N-Scale-2-S1-Shape-1-S1",
      "divider-1-offset-0",
      "divider-10-offset-0",
      "divider-100-offset-0",
    ],
    help="train/test split strategy",
    # dSpritres:
    #default="combinatorial2-Y-5-S3-X-5-S3-Orientation-4-N-Scale-1-S3-Shape-1-N", #richly diverse
    default="combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N", #poorly diverse
    #default="divider-1-offset-0"
  )
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

  if args.obverter_sampling_round_alternation_only:
    args.use_obverter_sampling = True 
  if args.use_aitao:
    assert 'similarity' in args.distractor_sampling

  if args.visual_context_consistent_obverter:
    args.context_consistent_obverter = True 
  if args.normalised_context_consistent_obverter:
    args.context_consistent_obverter = True

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
  transform = ResizeNormalize(
    size=stimulus_resize_dim, 
    normalize_rgb_values=normalize_rgb_values,
    rgb_scaler=rgb_scaler,
    use_cuda=False, #subprocess issue...s
  )

  from ReferentialGym.datasets.utils import AddEgocentricInvariance
  ego_inv_transform = AddEgocentricInvariance()

  transform_degrees = args.egocentric_tr_degrees
  transform_translate = (args.egocentric_tr_xy, args.egocentric_tr_xy)

  if args.with_descriptive_not_target_logit_language_conditioning:
    args.descriptive = True 

  default_descriptive_ratio = 1-(1/(args.nbr_train_distractors+2))
  # Default: 1-(1/(nbr_distractors+2)), 
  # otherwise the agent find the local minimum
  # where it only predicts "no-target"...
  if args.descriptive_ratio <=0.001:
    descriptive_ratio = default_descriptive_ratio
  else:
    descriptive_ratio = args.descriptive_ratio

  if args.obverter_threshold_to_stop_message_generation <= 0.0:
    if args.descriptive:
      args.obverter_threshold_to_stop_message_generation = 0.98
    else:
      nbr_category = 1 #target
      nbr_category += args.nbr_train_distractors
      #args.obverter_threshold_to_stop_message_generation = 1.9/nbr_category
      # v1:
      #args.obverter_threshold_to_stop_message_generation = 1.0-0.025*nbr_category
      # v1.1: increase the threshold halfway:
      args.obverter_threshold_to_stop_message_generation = 1.0-0.05*nbr_category

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
      "cultural_reset_meta_learning_rate":  1e-3,

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
      "adam_eps":                 1e-16,
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
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
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
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = 256 #args.symbol_processing_nbr_hidden_units
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
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = 50 #args.symbol_processing_nbr_hidden_units
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
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = 256 #args.symbol_processing_nbr_hidden_units
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
  
  elif "Baseline1CNN" in agent_config["architecture"]:
    rg_config["use_feat_converter"] = False
    agent_config["use_feat_converter"] = False
    
    if "BN" in args.arch:
      #agent_config["cnn_encoder_channels"] = ["BN32","BN32","BN32","BN32","BN32"]
      agent_config["cnn_encoder_channels"] = ["BN20","BN20","BN20","BN20","BN20"]
    else:
      agent_config["cnn_encoder_channels"] = [32,32,32,32,32]
    
    agent_config["cnn_encoder_kernels"] = [3,3,3,3,3]
    agent_config["cnn_encoder_strides"] = [2,2,2,2,2]
    #agent_config["cnn_encoder_paddings"] = [1,1,1,1,1]
    agent_config["cnn_encoder_paddings"] = [0,0,0,0,0]
    agent_config["cnn_encoder_non_linearities"] = [torch.nn.ReLU]
    agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
    # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
    
    # For a fair comparison between CNN an VAEs:
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = 50 #
    #agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
    #agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
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
    agent_config["cnn_encoder_feature_dim"] = args.agent_nbr_latent_dim
    #agent_config["cnn_encoder_feature_dim"] = args.symbol_processing_nbr_hidden_units
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
  
  elif "BetaVAE" in agent_config["architecture"]:
    make_VAE(agent_config, args, rg_config)
  else:
    raise NotImplementedError

  if "EncoderOnly" in agent_config["architecture"]:
    print("WARNING: Using Beta-VAE-EncoderOnly arch.")
    agent_config["architecture"] = "BN+" if "BN" in agent_config["architecture"] else ""
    agent_config["architecture"] += "Beta-VAE-EncoderOnlyCNN3x3"
  else:
    print("WARNING: Not using Beta-VAE-EncoderOnly arch.")

  baseline_vm_config = copy.deepcopy(agent_config)
  if args.with_baseline:
    if not("BetaVAE" in baseline_vm_config["architecture"]):
      print("WARNING: comparison of RG (without VAE) to VAE baseline.")
      baseline_vm_config["agent_architecture"] = "BN+BetaVAE3x3"
      baseline_vm_config["architecture"] = "BN+BetaVAE3x3"
      make_VAE(baseline_vm_config, args, {})
    else:
      print("WARNING: comparison of RG (+VAE) to VAE baseline.")


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
  
  if args.add_discriminative_test:
    save_path += f"WithDiscriminativeTest-{args.nbr_discriminative_test_distractors}Distractors/"
  if args.add_descr_discriminative_test:
    save_path += f"WithDescrDiscriminativeTest-{args.nbr_descr_discriminative_test_distractors}Distractors/"
  
  if args.add_descriptive_test:
    save_path += "WithDescriptiveTest/"

  if "synthetic" in args.graphtype:
    save_path += f"InterventionSyntheticCompositionalLanguage/{'WithObverterRoundAlternation/' if 'obverter' in args.graphtype else ''}"
    save_path += f"ProgressionEnd{args.synthetic_progression_end}/"
  
  if args.with_BN_in_obverter_decision_head:
    save_path += "DecisionHeadBN/"
  if args.with_DP_in_obverter_decision_head:
    save_path += "DecisionHeadDP0.5/"
    #save_path += "DecisionHeadDP0.2/"
    #save_path += "DecisionHeadDP0.1/"
  if args.with_DP_in_obverter_decision_head_listener_only:
    save_path += "ListenerDecisionHeadDP0.5Only/"
    #save_path += "ListenerDecisionHeadDP0.2Only/"
    #save_path += "ListenerDecisionHeadDP0.1Only/"
  if args.context_consistent_obverter:
    save_path += f"{'Visual' if args.visual_context_consistent_obverter else ''}{'Normalized' if args.normalised_context_consistent_obverter else ''}ContextConsistentObverter/"
  if args.with_descriptive_not_target_logit_language_conditioning:
    save_path += f"DescriptiveNotTargetLogicLanguageConditioning/"

  save_path += f"{args.dataset}+DualLabeled/AdamEPS{rg_config['adam_eps']}"
  if args.with_baseline:
    save_path += "WithBaselineArch/"

  if args.use_priority:
    save_path += "WithPrioritizedSampling/"
  if args.use_obverter_sampling:
    save_path += f"WithObverterSampling{'RoundAlternationOnly' if args.obverter_sampling_round_alternation_only else ''}/"

  if args.use_aitao:
    save_path += f"WithAITAO/"

  if args.egocentric:
    save_path += f"Egocentric-Rot{args.egocentric_tr_degrees}-XY{args.egocentric_tr_xy}/"
  
  save_path += f"/{nbr_epoch}Ep_Emb{rg_config['symbol_embedding_size']}"
  
  if not args.baseline_only:
    save_path += f"_CNN{cnn_feature_size}to{args.agent_nbr_latent_dim}"
    save_path += f"RNN{args.symbol_processing_nbr_hidden_units}"
  
  if args.with_baseline:
    save_path += f"/VAE{baseline_vm_config['architecture']}_CNN{args.vae_nbr_latent_dim}"
  
  if args.shared_architecture:
    save_path += "/shared_architecture"
  
  if not args.baseline_only:
    save_path += f"Dropout{rg_config['dropout_prob']}_DPEmb{rg_config['embedding_dropout_prob']}"
    save_path += f"_BN_{rg_config['agent_learning']}/"
    save_path += f"{rg_config['agent_loss_type']}"
  
  if 'dSprites' in args.dataset: 
    train_test_strategy = f"-{test_split_strategy}"
    if test_split_strategy != train_split_strategy:
      train_test_strategy = f"/train_{train_split_strategy}/test_{test_split_strategy}"
    save_path += f"/dSprites-split-{train_test_strategy}"
  
  elif '3DShapesPyBullet' in args.dataset: 
    train_test_strategy = f"-{train_split_strategy}"
    save_path += save_path_dataset+train_test_strategy
  
  if args.use_symbolic_stimuli:
    raise NotImplementedError
    rg_config["stimulus_resize_dim"] = 3 if 'Shapes' in args.dataset else 6 # dSprites...

  save_path += f"/OBS{rg_config['stimulus_resize_dim']}X{rg_config['stimulus_depth_dim']}C-Rep{rg_config['nbr_experience_repetition']}"
  
  if not args.baseline_only:
    if rg_config['use_curriculum_nbr_distractors']:
      save_path += f"+W{rg_config['curriculum_distractors_window_size']}Curr"
    if rg_config['with_utterance_penalization']:
      save_path += "+Tau-10-OOV{}PenProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
    if rg_config['with_utterance_promotion']:
      save_path += "+Tau-10-OOV{}ProProb{}".format(rg_config['utterance_factor'], rg_config['utterance_oov_prob'])  
    
  if rg_config['with_gradient_clip']:
    save_path += '+ClipGrad{}'.format(rg_config['gradient_clip'])
  
  if not args.baseline_only:
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
    
  #save_path += '-{}{}{}CulturalAgent-SEED{}-{}-obs_b{}_minib{}_lr{}-{}-tau0-{}-{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}'.\
  save_path += '-{}{}{}CulturalAgent-SEED{}-{}-obs_b{}_minib{}_lr{}-{}-tau0-{}/{}DistrTrain{}Test{}-stim{}-vocab{}over{}_{}{}/'.\
    format(
    'ObjectCentric' if rg_config['object_centric'] else '',
    'Descriptive{}'.format(rg_config['descriptive_target_ratio']) if rg_config['descriptive'] else '',
    'ContextConsistentObverter' if args.context_consistent_obverter else '',
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

  if 'BetaVAE' in baseline_vm_config['architecture']:
    save_path += "/BASELINE_VAE-"
    save_path +=f"{'Detached' if args.vae_detached_featout else ''}beta{vae_beta}-factor{factor_vae_gamma}" if 'BetaVAE' in baseline_vm_config['architecture'] else ''
  
  if 'MONet' in baseline_vm_config['architecture'] or 'BetaVAE' in baseline_vm_config['architecture']:
    save_path += f"beta{vae_beta}-factor{factor_vae_gamma}-gamma{monet_gamma}-sigma{vae_observation_sigma}" if 'MONet' in baseline_vm_config['architecture'] else ''
    save_path += f"CEMC{maxCap}over{nbrepochtillmaxcap}" if vae_constrainedEncoding else ''
    save_path += f"UnsupSeg{baseline_vm_config['unsupervised_segmentation_factor']}" if baseline_vm_config['unsupervised_segmentation_factor'] is not None else ''
    save_path += f"LossVAECoeff{args.vae_lambda}_{'UseMu' if args.vae_use_mu_value else ''}"

  if rg_config['use_feat_converter']:
    save_path += f"+FEATCONV"
  
  if rg_config['use_homoscedastic_multitasks_loss']:
    save_path += '+Homo'
  
  save_path += f"/{args.optimizer_type}/"


  if 'reinforce' in args.graphtype:
    save_path += f'/REINFORCE_EntropyCoeffNeg1m3/UnnormalizedDetLearningSignalHavrylovLoss/NegPG/'

  if 'obverter' in args.graphtype:
    save_path += f"{'ContextConsistent' if args.context_consistent_obverter else ''}Obverter{f'With{args.obverter_nbr_head_outputs}OututsDecisionHead' if args.obverter_use_decision_head and not(args.context_consistent_obverter) else ''}{args.obverter_threshold_to_stop_message_generation}-{args.obverter_nbr_games_per_round}GPR/DEBUG_{'OHE' if args.use_sentences_one_hot_vectors else ''}/"
  else:
    save_path += f"STGS-{args.agent_type}-LSTM-CNN-Agent/"

  save_path += "/WithSeparatedRoleMetric-AgentBased/DEBUG/"

  save_path += f"MetricPeriod{args.metric_epoch_period}+{f'b{args.metric_batch_size}' if args.metric_resampling else 'NO'}Resampling+DISComp-{'fast-' if args.metric_fast else ''}"#TestArchTanh/"
  save_path += f"{f'b{args.metric_batch_size}' if args.dis_metric_resampling else 'NO'}DisMetricResampling"
  save_path += f"+{'Active' if args.metric_active_factors_only else 'ALL'}Factors"
  save_path += "+NotNormalizedFeat+ResampleEncodeZ"
  save_path += f"_train{args.nbr_train_points}_test{args.nbr_eval_points}/"
  
  save_path += f'DatasetRepTrain{args.nbr_train_dataset_repetition}Test{args.nbr_test_dataset_repetition}'
  

  rg_config['save_path'] = save_path
  
  print(save_path)

  from ReferentialGym.utils import statsLogger
  logger = statsLogger(path=save_path,dumpPeriod=100)
  
  # # Agents
  batch_size = 4
  nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
  nbr_stimulus = agent_config['nbr_stimulus']
  
  obs_shape = [
    nbr_distractors+1,
    nbr_stimulus, 
    rg_config['stimulus_depth_dim'],
    rg_config['stimulus_resize_dim'],
    rg_config['stimulus_resize_dim']
  ]

  vocab_size = rg_config['vocab_size']
  max_sentence_length = rg_config['max_sentence_length']

  if not args.baseline_only:
      
    if args.context_consistent_obverter:
      from ReferentialGym.agents import ContextConsistentObverterAgent as ObverterAgent
    else:
      #from ReferentialGym.agents import DifferentiableObverterAgent
      #from ReferentialGym.agents.halfnew_differentiable_obverter_agent import DifferentiableObverterAgent
      from ReferentialGym.agents.obverter_agent import ObverterAgent
      #from ReferentialGym.agents.depr_differentiable_obverter_agent import DifferentiableObverterAgent
    
    if 'obverter' in args.graphtype:
      """
      speaker = DifferentiableObverterAgent(
        kwargs=agent_config, 
        obs_shape=obs_shape, 
        vocab_size=vocab_size, 
        max_sentence_length=max_sentence_length,
        agent_id='s0',
        logger=logger,
        use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
        use_decision_head_=args.obverter_use_decision_head,
        nbr_head_outputs=args.obverter_nbr_head_outputs,
        differentiable=args.differentiable
      )
      """
      speaker = ObverterAgent(
        kwargs=agent_config, 
        obs_shape=obs_shape, 
        vocab_size=vocab_size, 
        max_sentence_length=max_sentence_length,
        agent_id='s0',
        logger=logger,
        use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
        use_language_projection=args.visual_context_consistent_obverter,
        use_normalized_scores=args.normalised_context_consistent_obverter,
        with_BN_in_decision_head=args.with_BN_in_obverter_decision_head,
        with_DP_in_decision_head=args.with_DP_in_obverter_decision_head,
        with_DP_in_listener_decision_head_only=args.with_DP_in_obverter_decision_head_listener_only,
        with_descriptive_not_target_logit_language_conditioning=args.with_descriptive_not_target_logit_language_conditioning,
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
    
    obs_shape = [
      nbr_distractors+1,
      nbr_stimulus, 
      rg_config['stimulus_depth_dim'],
      rg_config['stimulus_resize_dim'],
      rg_config['stimulus_resize_dim']
    ]

    vocab_size = rg_config['vocab_size']
    max_sentence_length = rg_config['max_sentence_length']

    if 'obverter' in args.graphtype:
      """
      listener = DifferentiableObverterAgent(
        kwargs=listener_config, 
        obs_shape=obs_shape, 
        vocab_size=vocab_size, 
        max_sentence_length=max_sentence_length,
        agent_id='l0',
        logger=logger,
        use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
        use_decision_head_=args.obverter_use_decision_head,
        nbr_head_outputs=args.obverter_nbr_head_outputs,
        differentiable=args.differentiable
      )
      """
      listener = ObverterAgent(
        kwargs=listener_config, 
        obs_shape=obs_shape, 
        vocab_size=vocab_size, 
        max_sentence_length=max_sentence_length,
        agent_id='l0',
        logger=logger,
        use_sentences_one_hot_vectors=args.use_sentences_one_hot_vectors,
        use_language_projection=args.visual_context_consistent_obverter,
        with_BN_in_decision_head=args.with_BN_in_obverter_decision_head,
        with_DP_in_decision_head=args.with_DP_in_obverter_decision_head,
        with_DP_in_listener_decision_head_only=args.with_DP_in_obverter_decision_head_listener_only,
        with_descriptive_not_target_logit_language_conditioning=args.with_descriptive_not_target_logit_language_conditioning,
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
  if not args.baseline_only:
    modules[speaker.id] = speaker 
    modules[listener.id] = listener 

  if "synthetic" in args.graphtype:
    intervention_id = "intervention_0"
    intervention_config = {
      "output_ohe_placeholder": "modules:current_speaker:sentences_one_hot",
      "output_widx_placeholder": "modules:current_speaker:sentences_widx",
      "vocab_size": rg_config["vocab_size"],
      "max_sentence_length": rg_config["max_sentence_length"],
      "vocab_stop_idx":listener.vocab_stop_idx,
      "epoch_progression_end":args.synthetic_progression_end,
    }
    intervention_stream_ids = {
      "logs_dict":"logs_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",
      "input":"current_dataloader:sample:speaker_exp_latents_values", 
      "sentences_one_hot":"modules:current_speaker:sentences_one_hot",
      "sentences_widx":"modules:current_speaker:sentences_widx",
    }
    modules[intervention_id] = InterventionModule(
      id=intervention_id,
      config=intervention_config,
      input_stream_ids=intervention_stream_ids
    )
    
  from ReferentialGym import modules as rg_modules

  # Sampler:
  if args.use_obverter_sampling:
    obverter_sampling_id = "obverter_sampling_0"
    obverter_sampling_config = {
      "batch_size": rg_config["batch_size"],
      "round_alternation_only": args.obverter_sampling_round_alternation_only,
      "obverter_nbr_games_per_round": args.obverter_nbr_games_per_round,
    }
  
  if args.use_aitao:
    aitao_id = "aitao_0"
    aitao_config = {}


  if not args.baseline_only:
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

  if args.use_obverter_sampling:
    modules[obverter_sampling_id] = rg_modules.ObverterDatasamplingModule(
      id=obverter_sampling_id,
      config=obverter_sampling_config,
    )
  
  if args.use_aitao:
    from aitao_module import AITAOModule
    modules[aitao_id] = AITAOModule(
      id=aitao_id,
      config=aitao_config,
    )
  
 
  if not args.baseline_only:
    modules[population_handler_id] = rg_modules.build_PopulationHandlerModule(
        id=population_handler_id,
        prototype_speaker=speaker,
        prototype_listener=listener,
        config=population_handler_config,
        input_stream_ids=population_handler_stream_ids)

    modules[current_speaker_id] = rg_modules.CurrentAgentModule(id=current_speaker_id,role="speaker")
    modules[current_listener_id] = rg_modules.CurrentAgentModule(id=current_listener_id,role="listener")
    
  # Baseline:
  if args.with_baseline:  
    obs_shape = [nbr_distractors+1,nbr_stimulus, rg_config['stimulus_depth_dim'],rg_config['stimulus_resize_dim'],rg_config['stimulus_resize_dim']]
    baseline_vm_config['obs_shape'] = obs_shape
    
    baseline_vm_id = f"baseline_{baseline_vm_config['architecture']}"
    baseline_vm_input_stream_ids = {
      "losses_dict":"losses_dict",
      "logs_dict":"logs_dict",
      "mode":"signals:mode",
      "inputs":"current_dataloader:sample:speaker_experiences",
    }

    modules[baseline_vm_id] = rg_modules.build_VisualModule(
      id=baseline_vm_id, 
      config=baseline_vm_config,
      input_stream_ids=baseline_vm_input_stream_ids
    )

    baseline_vm_latent_traversal_id = f"baseline_{baseline_vm_config['architecture']}_latent_traversal_query"
    baseline_vm_latent_traversal_input_stream_ids = {
      "logger":"modules:logger:ref",
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

      "model":f"modules:{baseline_vm_id}:ref",
      "experiences":"current_dataloader:sample:speaker_experiences", 
    }

    modules[baseline_vm_latent_traversal_id] = rg_modules.build_VAELatentTraversalQueryModule(
      id=baseline_vm_latent_traversal_id, 
      config={
        "epoch_period":args.metric_epoch_period,
        "traversal": False,
      },
      input_stream_ids=baseline_vm_latent_traversal_input_stream_ids,
    )


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

  if not args.baseline_only:
    """
    topo_sim_metric_id = "topo_sim_metric"
    topo_sim_metric_module = rg_modules.build_TopographicSimilarityMetricModule(
      id=topo_sim_metric_id,
      config = {
        "parallel_TS_computation_max_workers":args.parallel_TS_worker,
        "epoch_period":args.metric_epoch_period,
        "fast":args.metric_fast,
        "verbose":False,
        "vocab_size":rg_config["vocab_size"],
      }
    )
    modules[topo_sim_metric_id] = topo_sim_metric_module
    """
    speaker_topo_sim_metric_id = f"{speaker.id}_topo_sim2_metric"
    speaker_topo_sim_metric_input_stream_ids = {
      #"model":"modules:current_speaker:ref:ref_agent",
      "model":f"modules:{speaker.id}:ref:_utter",
      "features":"modules:current_speaker:ref:ref_agent:features",
      "representations":"modules:current_speaker:sentences_widx",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
      "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
      "indices":"current_dataloader:sample:speaker_indices", 
    }
    
    def agent_preprocess_fn(x):
      if args.use_cuda:
        x = x.cuda()
        # adding distractor x stimuli-dim dims:
      x = x.unsqueeze(1).unsqueeze(1)
      return x 

    def agent_postprocess_fn(x):
      x = x[1].cpu().detach()
      x = x.reshape((x.shape[0],-1)).numpy()
      return x 

    def agent_features_postprocess_fn(x):
      x = x[-1].cpu().detach()
      x = x.reshape((x.shape[0],-1)).numpy()
      return x 

    speaker_topo_sim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
      id=speaker_topo_sim_metric_id,
      config = {
        "metric_fast": args.metric_fast,
        "pvalue_significance_threshold": 0.05,
        "parallel_TS_computation_max_workers":args.parallel_TS_worker,
        "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
        # cf outputs of _utter:
        "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
        # not necessary if providing a preprocess_fn, 
        # that computes the features/_sense output, but here it is in order to deal with shapes:
        "features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
        #"preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        # cf _sense:
        "preprocess_fn": (lambda x: speaker._sense(agent_preprocess_fn(x))),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      },
      input_stream_ids=speaker_topo_sim_metric_input_stream_ids,
    )
    modules[speaker_topo_sim_metric_id] = speaker_topo_sim_metric_module

    posbosdis_disentanglement_metric_id = "posbosdis_disentanglement_metric"
    posbosdis_disentanglement_metric_input_stream_ids = {
      #"model":"modules:current_speaker:ref:ref_agent",
      "model":"modules:current_speaker:ref:ref_agent:_utter",
      "representations":"modules:current_speaker:sentences_widx",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      #"latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
      "indices":"current_dataloader:sample:speaker_indices", 
    }

    posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
      id=posbosdis_disentanglement_metric_id,
      input_stream_ids=posbosdis_disentanglement_metric_input_stream_ids,
      config = {
        "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[posbosdis_disentanglement_metric_id] = posbosdis_disentanglement_metric_module

    speaker_posbosdis_metric_id = f"{speaker.id}_posbosdis_disentanglement_metric"
    speaker_posbosdis_input_stream_ids = {
      #"model":"modules:current_speaker:ref:ref_agent",
      "model":f"modules:{speaker.id}:ref:_utter",
      "representations":"modules:current_speaker:sentences_widx",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
      "indices":"current_dataloader:sample:speaker_indices", 
    }
    
    def agent_preprocess_fn(x):
      if args.use_cuda:
        x = x.cuda()
        # adding distractor x stimuli-dim dims:
      x = x.unsqueeze(1).unsqueeze(1)
      return x 

    def agent_postprocess_fn(x):
      x = x[1].cpu().detach()
      x = x.reshape((x.shape[0],-1)).numpy()
      return x 

    speaker_posbosdis_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
      id=speaker_posbosdis_metric_id,
      config = {
        "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
        # cf outputs of _utter:
        "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
        #"preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        # cf _sense:
        "preprocess_fn": (lambda x: speaker._sense(agent_preprocess_fn(x))),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      },
      input_stream_ids=speaker_posbosdis_input_stream_ids,
    )
    modules[speaker_posbosdis_metric_id] = speaker_posbosdis_metric_module

    if "obverter" in args.graphtype:
      listener_topo_sim_metric_id = f"{listener.id}_topo_sim2_metric"
      listener_topo_sim_metric_input_stream_ids = {
        #"model":"modules:current_speaker:ref:ref_agent",
        "model":f"modules:{listener.id}:ref:_utter",
        "features":"modules:current_speaker:ref:ref_agent:features",
        "representations":"modules:current_speaker:sentences_widx",
        "experiences":"current_dataloader:sample:speaker_experiences", 
        "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
        "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
        "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
        "indices":"current_dataloader:sample:speaker_indices", 
      }
      
      listener_topo_sim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
        id=listener_topo_sim_metric_id,
        config = {
          "pvalue_significance_threshold": 0.5,
          "parallel_TS_computation_max_workers":args.parallel_TS_worker,
          "filtering_fn":(lambda kwargs: listener.role=="speaker"),
          #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
          # cf outputs of _utter:
          "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
          # not necessary if providing a preprocess_fn, 
          # that computes the features/_sense output, but here it is:
          "features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
          #"preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
          # cf _sense:
          "preprocess_fn": (lambda x: listener._sense(agent_preprocess_fn(x))),
          "epoch_period":args.metric_epoch_period,
          "batch_size":args.metric_batch_size,#5,
          "nbr_train_points":args.nbr_train_points,#3000,
          "nbr_eval_points":args.nbr_eval_points,#2000,
          "resample": args.metric_resampling,
          "threshold":5e-2,#0.0,#1.0,
          "random_state_seed":args.seed,
          "verbose":False,
          "active_factors_only":args.metric_active_factors_only,
        },
        input_stream_ids=listener_topo_sim_metric_input_stream_ids,
      )
      modules[listener_topo_sim_metric_id] = listener_topo_sim_metric_module

      listener_posbosdis_metric_id = f"{listener.id}_posbosdis_disentanglement_metric"
      listener_posbosdis_input_stream_ids = {
        #"model":"modules:current_speaker:ref:ref_agent",
        "model":f"modules:{listener.id}:ref:_utter",
        "representations":"modules:current_speaker:sentences_widx",
        "experiences":"current_dataloader:sample:speaker_experiences", 
        "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
        "indices":"current_dataloader:sample:speaker_indices", 
      }
      
      listener_posbosdis_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
        id=listener_posbosdis_metric_id,
        config = {
          "filtering_fn":(lambda kwargs: listener.role=="speaker"),
          #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
          # cf outputs of _utter:
          "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
          #"preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
          # cf _sense:
          "preprocess_fn": (lambda x: listener._sense(agent_preprocess_fn(x))),
          "epoch_period":args.metric_epoch_period,
          "batch_size":args.metric_batch_size,#5,
          "nbr_train_points":args.nbr_train_points,#3000,
          "nbr_eval_points":args.nbr_eval_points,#2000,
          "resample": args.metric_resampling,
          "threshold":5e-2,#0.0,#1.0,
          "random_state_seed":args.seed,
          "verbose":False,
          "active_factors_only":args.metric_active_factors_only,
        },
        input_stream_ids=listener_posbosdis_input_stream_ids,
      )
      modules[listener_posbosdis_metric_id] = listener_posbosdis_metric_module



    # Modularity: Speaker
    speaker_modularity_disentanglement_metric_id = f"{speaker.id}_modularity_disentanglement_metric"
    speaker_modularity_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{speaker.id}:ref:cnn_encoder",
      "representations":f"modules:{speaker.id}:ref:features",
      "experiences":f"modules:{speaker.id}:ref:experiences", 
      "latent_representations":f"modules:{speaker.id}:ref:exp_latents", 
      "indices":f"modules:{speaker.id}:ref:indices", 
    }
    speaker_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
      id=speaker_modularity_disentanglement_metric_id,
      input_stream_ids=speaker_modularity_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        # dealing with extracting z (mu in pos 1):
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[speaker_modularity_disentanglement_metric_id] = speaker_modularity_disentanglement_metric_module

    # Modularity: Listener
    listener_modularity_disentanglement_metric_id = f"{listener.id}_modularity_disentanglement_metric"
    listener_modularity_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{listener.id}:ref:cnn_encoder",
      "representations":f"modules:{listener.id}:ref:features",
      "experiences":f"modules:{listener.id}:ref:experiences", 
      "latent_representations":f"modules:{listener.id}:ref:exp_latents", 
      "indices":f"modules:{listener.id}:ref:indices", 
    }
    listener_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
      id=listener_modularity_disentanglement_metric_id,
      input_stream_ids=listener_modularity_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: listener.role=="speaker"),
        #"filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[listener_modularity_disentanglement_metric_id] = listener_modularity_disentanglement_metric_module
    
    inst_coord_metric_id = f"inst_coord_metric"
    inst_coord_input_stream_ids = {
      "logger":"modules:logger:ref",
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
      "dataset":"current_dataset:ref",

      "vocab_size":"config:vocab_size",
      "max_sentence_length":"config:max_sentence_length",
      "sentences_widx":"modules:current_speaker:sentences_widx", 
      "decision_probs":"modules:current_listener:decision_probs",
      "listener_indices":"current_dataloader:sample:listener_indices",
    }
    inst_coord_metric_module = rg_modules.build_InstantaneousCoordinationMetricModule(
      id=inst_coord_metric_id,
      config = {
        "filtering_fn":(lambda kwargs: True),
        "epoch_period":1,
      },
      input_stream_ids=inst_coord_input_stream_ids,
    )
    modules[inst_coord_metric_id] = inst_coord_metric_module
    
    speaker_inst_coord_metric_id = f"{speaker.id}_inst_coord_metric"
    speaker_inst_coord_input_stream_ids = {
      "logger":"modules:logger:ref",
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
      "dataset":"current_dataset:ref",

      "vocab_size":"config:vocab_size",
      "max_sentence_length":"config:max_sentence_length",
      "sentences_widx":"modules:current_speaker:sentences_widx", 
      "decision_probs":"modules:current_listener:decision_probs",
      "listener_indices":"current_dataloader:sample:listener_indices",
    }
    speaker_inst_coord_metric_module = rg_modules.build_InstantaneousCoordinationMetricModule(
      id=speaker_inst_coord_metric_id,
      config = {
        "filtering_fn":(lambda kwargs: speaker.role=="listener"),
        "epoch_period":1,
      },
      input_stream_ids=speaker_inst_coord_input_stream_ids,
    )
    modules[speaker_inst_coord_metric_id] = speaker_inst_coord_metric_module

    # Listener agent:
    listener_inst_coord_metric_id = f"{listener.id}_inst_coord_metric"
    listener_inst_coord_input_stream_ids = {
      "logger":"modules:logger:ref",
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
      "dataset":"current_dataset:ref",

      "vocab_size":"config:vocab_size",
      "max_sentence_length":"config:max_sentence_length",
      "sentences_widx":"modules:current_speaker:sentences_widx", 
      "decision_probs":"modules:current_listener:decision_probs",
      "listener_indices":"current_dataloader:sample:listener_indices",
    }
    listener_inst_coord_metric_module = rg_modules.build_InstantaneousCoordinationMetricModule(
      id=listener_inst_coord_metric_id,
      config = {
        "filtering_fn":(lambda kwargs: listener.role=="listener"),
        "epoch_period":1,
      },
      input_stream_ids=listener_inst_coord_input_stream_ids,
    )
    modules[listener_inst_coord_metric_id] = listener_inst_coord_metric_module
  

  if 'dSprites' in args.dataset:
    pass
    """
    if not args.baseline_only:
      dsprites_latent_metric_id = "dsprites_latent_metric"
      dsprites_latent_metric_module = rg_modules.build_dSpritesPerLatentAccuracyMetricModule(id=dsprites_latent_metric_id,
        config = {
          "epoch_period":1,
        }
      )
      modules[dsprites_latent_metric_id] = dsprites_latent_metric_module
    """
    """
    if args.with_baseline:
      raise NotImplementedError
      # TODO: implement baseline_mhcm
      baseline_dsprites_latent_metric_id = "baseline_dsprites_latent_metric"
      baseline_dsprites_input_streams_ids = {
        "logger":"modules:logger:ref",
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
        
        "accuracy":f"modules:{baseline_mhcm}:accuracy",
        "test_latents_mask":"current_dataloader:sample:speaker_exp_test_latents_masks",
      }
      baseline_dsprites_latent_metric_module = rg_modules.build_dSpritesPerLatentAccuracyMetricModule(
        id=baseline_dsprites_latent_metric_id,
        config = {
          "epoch_period":1,
        },
        input_stream_ids=baseline_vm_input_stream_ids
      )
      modules[baseline_dsprites_latent_metric_id] = baseline_dsprites_latent_metric_module
    """
  if not args.baseline_only:
    speaker_factor_vae_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{speaker.id}:ref:cnn_encoder",
      "representations":f"modules:{speaker.id}:ref:features",
      "experiences":f"modules:{speaker.id}:ref:experiences", 
      "latent_representations":f"modules:{speaker.id}:ref:exp_latents", 
      "latent_values_representations":f"modules:{speaker.id}:ref:exp_latents_values",
      "indices":f"modules:{speaker.id}:ref:indices", 
    }
    speaker_factor_vae_disentanglement_metric_id = f"{speaker.id}_factor_vae_disentanglement_metric"
    speaker_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
      id=speaker_factor_vae_disentanglement_metric_id,
      input_stream_ids=speaker_factor_vae_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: speaker.role=="speaker"),
        #"filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points": args.nbr_train_points,#3000,
        "nbr_eval_points": args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[speaker_factor_vae_disentanglement_metric_id] = speaker_factor_vae_disentanglement_metric_module

    listener_factor_vae_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{listener.id}:ref:cnn_encoder",
      "representations":f"modules:{listener.id}:ref:features",
      "experiences":f"modules:{listener.id}:ref:experiences", 
      "latent_representations":f"modules:{listener.id}:ref:exp_latents", 
      "latent_values_representations":f"modules:{listener.id}:ref:exp_latents_values",
      "indices":f"modules:{listener.id}:ref:indices", 
    }
    listener_factor_vae_disentanglement_metric_id = f"{listener.id}_factor_vae_disentanglement_metric"
    listener_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
      id=listener_factor_vae_disentanglement_metric_id,
      input_stream_ids=listener_factor_vae_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: listener.role=="speaker"),
        #"filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points": args.nbr_train_points,#3000,
        "nbr_eval_points": args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[listener_factor_vae_disentanglement_metric_id] = listener_factor_vae_disentanglement_metric_module
    
    # Mutual Information Gap:
    speaker_mig_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{speaker.id}:ref:cnn_encoder",
      "representations":f"modules:{speaker.id}:ref:features",
      "experiences":f"modules:{speaker.id}:ref:experiences", 
      "latent_representations":f"modules:{speaker.id}:ref:exp_latents", 
      "indices":f"modules:{speaker.id}:ref:indices", 
    }
    speaker_mig_disentanglement_metric_id = f"{speaker.id}_mig_disentanglement_metric"
    speaker_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
      id=speaker_mig_disentanglement_metric_id,
      input_stream_ids=speaker_mig_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: speaker.role=="speaker"),
        #"filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample":args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[speaker_mig_disentanglement_metric_id] = speaker_mig_disentanglement_metric_module

    listener_mig_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{listener.id}:ref:cnn_encoder",
      "representations":f"modules:{listener.id}:ref:features",
      "experiences":f"modules:{listener.id}:ref:experiences", 
      "latent_representations":f"modules:{listener.id}:ref:exp_latents", 
      "indices":f"modules:{listener.id}:ref:indices", 
    }
    listener_mig_disentanglement_metric_id = f"{listener.id}_mig_disentanglement_metric"
    listener_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
      id=listener_mig_disentanglement_metric_id,
      input_stream_ids=listener_mig_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: listener.role=="speaker"),
        #"filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample":args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[listener_mig_disentanglement_metric_id] = listener_mig_disentanglement_metric_module

  # Mutual Information Gap:
  if args.with_baseline:
    baseline_factor_vae_disentanglement_metric_id = "baseline_factor_vae_disentanglement_metric"
    baseline_factor_vae_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{baseline_vm_id}:ref:encoder",
      # Retrieve the function that outputs Z directly:
      #"model":f"modules:{baseline_vm_id}:ref:encoder:encodeZ",
      "representations":f"modules:{baseline_vm_id}:ref:features_not_normalized",
      #"representations":f"modules:{baseline_vm_id}:ref:features",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
      "indices":"current_dataloader:sample:speaker_indices", 
    }
    baseline_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
      id=baseline_factor_vae_disentanglement_metric_id,
      input_stream_ids=baseline_factor_vae_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        #"postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        # Sampling Z, assuming model has method encodeZ :
        "postprocess_fn": (lambda x: x[0].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        # Sampling Mu, assuming model has method encodeZ :
        #"postprocess_fn": (lambda x: x[1].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        # Sampling Mu||Logvar, assuming model has method encodeZ :
        #"postprocess_fn": (lambda x: torch.cat([x[1],x[2]], dim=-1).cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points": args.nbr_train_points,#3000,
        "nbr_eval_points": args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[baseline_factor_vae_disentanglement_metric_id] = baseline_factor_vae_disentanglement_metric_module
    
    baseline_modularity_disentanglement_metric_id = "baseline_modularity_disentanglement_metric"
    baseline_modularity_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{baseline_vm_id}:ref:encoder",
      # Retrieve the function that outputs Z directly:
      #"model":f"modules:{baseline_vm_id}:ref:encoder:encodeZ",
      "representations":f"modules:{baseline_vm_id}:ref:features_not_normalized",
      #"representations":f"modules:{baseline_vm_id}:ref:features",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      "indices":"current_dataloader:sample:speaker_indices", 
    }
    baseline_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
      id=baseline_modularity_disentanglement_metric_id,
      input_stream_ids=baseline_modularity_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        #"postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        # Sampling Z, assuming model has method encodeZ :
        "postprocess_fn": (lambda x: x[0].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample": args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[baseline_modularity_disentanglement_metric_id] = baseline_modularity_disentanglement_metric_module
    baseline_mig_disentanglement_metric_id = "baseline_mig_disentanglement_metric"
    baseline_mig_disentanglement_metric_input_stream_ids = {
      "model":f"modules:{baseline_vm_id}:ref:encoder",
      # Retrieve the function that outputs Z directly:
      #"model":f"modules:{baseline_vm_id}:ref:encoder:encodeZ",
      "representations":f"modules:{baseline_vm_id}:ref:features_not_normalized",
      #"representations":f"modules:{baseline_vm_id}:ref:features",
      "experiences":"current_dataloader:sample:speaker_experiences", 
      "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
      "indices":"current_dataloader:sample:speaker_indices", 
    }
    baseline_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
      id=baseline_mig_disentanglement_metric_id,
      input_stream_ids=baseline_mig_disentanglement_metric_input_stream_ids,
      config = {
        "filtering_fn": (lambda kwargs: True),
        #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
        #"postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        # Sampling Z, assuming model has method encodeZ :
        "postprocess_fn": (lambda x: x[0].cpu().detach().numpy() if "BetaVAE" in baseline_vm_config["architecture"] else x.cpu().detach().numpy()),
        "preprocess_fn": (lambda x: x.cuda() if args.use_cuda else x),
        "epoch_period":args.metric_epoch_period,
        "batch_size":args.metric_batch_size,#5,
        "nbr_train_points":args.nbr_train_points,#3000,
        "nbr_eval_points":args.nbr_eval_points,#2000,
        "resample":args.dis_metric_resampling,
        "threshold":5e-2,#0.0,#1.0,
        "random_state_seed":args.seed,
        "verbose":False,
        "active_factors_only":args.metric_active_factors_only,
      }
    )
    modules[baseline_mig_disentanglement_metric_id] = baseline_mig_disentanglement_metric_module


  logger_id = "per_epoch_logger"
  logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
  modules[logger_id] = logger_module

  if not args.baseline_only:
    pipelines["referential_game"] = [population_handler_id]
  
    if args.use_obverter_sampling:
      pipelines["referential_game"].append(obverter_sampling_id)
     
    if "synthetic" in args.graphtype:
      pipelines["referential_game"] += [
        current_speaker_id,
        intervention_id,
        current_listener_id
      ]
    else:
      pipelines["referential_game"] += [
        current_speaker_id,
        current_listener_id
      ]
    
    if args.use_aitao:
      pipelines["referential_game"].append(aitao_id)
    

  if args.with_baseline:
    pipelines[baseline_vm_id] = []

    if args.use_obverter_sampling and args.baseline_only:
      pipelines[baseline_vm_id] += [obverter_sampling_id]
    
    pipelines[baseline_vm_id] += [baseline_vm_id]

  pipelines[optim_id] = []
  if args.homoscedastic_multitasks_loss:
    pipelines[optim_id].append(homo_id)
  pipelines[optim_id].append(optim_id)
  """
  # Add gradient recorder module for debugging purposes:
  pipelines[optim_id].append(grad_recorder_id)
  """
  if not args.baseline_only:
    pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
    pipelines[optim_id].append(listener_modularity_disentanglement_metric_id)
    pipelines[optim_id].append(listener_mig_disentanglement_metric_id)
    if not(args.shared_architecture):
      pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
      pipelines[optim_id].append(speaker_modularity_disentanglement_metric_id)
      pipelines[optim_id].append(speaker_mig_disentanglement_metric_id)
    
  if args.with_baseline:
    pipelines[optim_id].append(baseline_factor_vae_disentanglement_metric_id)
    pipelines[optim_id].append(baseline_modularity_disentanglement_metric_id)
    pipelines[optim_id].append(baseline_mig_disentanglement_metric_id)

  if not args.baseline_only:
    #pipelines[optim_id].append(topo_sim_metric_id)
    pipelines[optim_id].append(speaker_topo_sim_metric_id)
    #pipelines[optim_id].append(posbosdis_disentanglement_metric_id)
    pipelines[optim_id].append(speaker_posbosdis_metric_id)
    if "obverter" in args.graphtype:
      pipelines[optim_id].append(listener_topo_sim_metric_id)
      pipelines[optim_id].append(listener_posbosdis_metric_id)
    pipelines[optim_id].append(inst_coord_metric_id)
    pipelines[optim_id].append(speaker_inst_coord_metric_id)
    pipelines[optim_id].append(listener_inst_coord_metric_id)
    
  """
  if args.with_baseline:
    pipelines[optim_id].append(baseline_vm_latent_traversal_id)
  """

  """
  if 'dSprites' in args.dataset \
  and not args.baseline_only:  
    pipelines[optim_id].append(dsprites_latent_metric_id)

    if args.with_baseline:
      pipelines[optim_id].append(baseline_dsprites_latent_metric)      
  """

  pipelines[optim_id].append(logger_id)

  rg_config["modules"] = modules
  rg_config["pipelines"] = pipelines


  # dataset_args = {
  #     "dataset_class":            "DualLabeledDataset",
  #     "modes": {"train": train_dataset,
  #               "test": test_dataset,
  #               },
  #     "need_dict_wrapping":       need_dict_wrapping,
  #     "nbr_stimulus":             rg_config["nbr_stimulus"],
  #     "distractor_sampling":      rg_config["distractor_sampling"],
  #     "nbr_distractors":          rg_config["nbr_distractors"],
  #     "observability":            rg_config["observability"],
  #     "object_centric":           rg_config["object_centric"],
  #     "descriptive":              rg_config["descriptive"],
  #     "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
  # }
  dataset_args = {"modes":["train", "test"]}
  dataset_args["train"] = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {
        "train": train_dataset,
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
  dataset_args["test"] = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {
        "train": train_dataset,
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

  if args.add_descriptive_test:
    dataset_args["modes"].append("descriptive_test")
    nbd = {"descriptive_test":0}
    nbd.update(rg_config["nbr_distractors"])
    dataset_args["descriptive_test"] = {
        "dataset_class":            "DualLabeledDataset",
        "modes": {
          "train": train_dataset,
          "descriptive_test": test_dataset,
        },
        "need_dict_wrapping":       need_dict_wrapping,
        "nbr_stimulus":             rg_config["nbr_stimulus"],
        "distractor_sampling":      rg_config["distractor_sampling"],
        "nbr_distractors":          nbd,
        "observability":            rg_config["observability"],
        "object_centric":           rg_config["object_centric"],
        "descriptive":              True, #rg_config["descriptive"],
        "descriptive_target_ratio": 0.5, #rg_config["descriptive_target_ratio"],
    }  
  if args.add_discriminative_test:
    dataset_args["modes"].append("discriminative_test")
    nbd = {"discriminative_test":args.nbr_discriminative_test_distractors}
    nbd.update(rg_config["nbr_distractors"])
    dataset_args["discriminative_test"] = {
        "dataset_class":            "DualLabeledDataset",
        "modes": {
          "train": train_dataset,
          "discriminative_test": test_dataset,
        },
        "need_dict_wrapping":       need_dict_wrapping,
        "nbr_stimulus":             rg_config["nbr_stimulus"],
        "distractor_sampling":      rg_config["distractor_sampling"],
        "nbr_distractors":          nbd,
        "observability":            rg_config["observability"],
        "object_centric":           rg_config["object_centric"],
        "descriptive":              False, #rg_config["descriptive"],
        "descriptive_target_ratio": 1.0, #rg_config["descriptive_target_ratio"],
    }

    dataset_args["modes"].append("discriminative_validation_test")
    nbd = {"discriminative_validation_test":args.nbr_discriminative_test_distractors}
    nbd.update(rg_config["nbr_distractors"])
    dataset_args["discriminative_validation_test"] = {
        "dataset_class":            "DualLabeledDataset",
        "modes": {
          "train": train_dataset,
          "discriminative_validation_test": train_dataset,
        },
        "need_dict_wrapping":       need_dict_wrapping,
        "nbr_stimulus":             rg_config["nbr_stimulus"],
        "distractor_sampling":      rg_config["distractor_sampling"],
        "nbr_distractors":          nbd,
        "observability":            rg_config["observability"],
        "object_centric":           rg_config["object_centric"],
        "descriptive":              False, #rg_config["descriptive"],
        "descriptive_target_ratio": 1.0, #rg_config["descriptive_target_ratio"],
    }  

  if args.add_descr_discriminative_test and rg_config["descriptive"]:
    dataset_args["modes"].append("descr_discriminative_test")
    nbd = {"descr_discriminative_test":args.nbr_descr_discriminative_test_distractors}
    nbd.update(rg_config["nbr_distractors"])
    dataset_args["descr_discriminative_test"] = {
        "dataset_class":            "DualLabeledDataset",
        "modes": {
          "train": train_dataset,
          "descr_discriminative_test": test_dataset,
        },
        "need_dict_wrapping":       need_dict_wrapping,
        "nbr_stimulus":             rg_config["nbr_stimulus"],
        "distractor_sampling":      rg_config["distractor_sampling"],
        "nbr_distractors":          nbd,
        "observability":            rg_config["observability"],
        "object_centric":           rg_config["object_centric"],
        "descriptive":              rg_config["descriptive"],
        "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
    }

    dataset_args["modes"].append("descr_discriminative_validation_test")
    nbd = {"descr_discriminative_validation_test":args.nbr_descr_discriminative_test_distractors}
    nbd.update(rg_config["nbr_distractors"])
    dataset_args["descr_discriminative_validation_test"] = {
        "dataset_class":            "DualLabeledDataset",
        "modes": {
          "train": train_dataset,
          "descr_discriminative_validation_test": train_dataset,
        },
        "need_dict_wrapping":       need_dict_wrapping,
        "nbr_stimulus":             rg_config["nbr_stimulus"],
        "distractor_sampling":      rg_config["distractor_sampling"],
        "nbr_distractors":          nbd,
        "observability":            rg_config["observability"],
        "object_centric":           rg_config["object_centric"],
        "descriptive":              rg_config["descriptive"],
        "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
    }  

  rg_config['use_priority'] = args.use_priority

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
