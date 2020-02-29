import random
import numpy as np
import cv2 

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
  nbr_epoch = 100
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
      "agent_learning":           'transfer_learning',  #'transfer_learning' : CNN's outputs are detached from the graph...
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

      "batch_size":               128, #64
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

  # Super Aggressive:
  #train_split_strategy = 'combinatorial3-1FPY-16-2-X-16-2-Orientation-10-4-Scale-1-5-Shape-1-3'
  #train_split_strategy = 'combinatorial3-0FPY-8-4-X-8-4-Orientation-10-4-Scale-1-5-0FP_Shape-1-3'
  
  # Not creating test set in some latent axis:
  # Normal (undensified orientation...):
  #train_split_strategy = 'combinatorial3-Y-1-5-X-1-5-Orientation-10-N-Scale-1-N-1FP_Shape-1-N'
  # Undensified:
  #train_split_strategy = 'combinatorial3-Y-2-5-X-2-5-Orientation-10-N-Scale-2-N-1FP_Shape-1-N'
  # Aggressively Undensified:
  #train_split_strategy = 'combinatorial3-Y-4-2-X-4-2-Orientation-10-N-Scale-2-N-0FP_Shape-1-N'
  #train_split_strategy = 'combinatorial3-Y-4-2-X-4-2-Orientation-10-N-Scale-2-N-Shape-1-3'
  # Agressive: idem + test values on each axis: 'primitive around right' 0-filler context
  #Not enough test data: train_split_strategy = 'combinatorial5-Y-4-2-X-4-2-Orientation-10-3-Scale-2-2-0FP_Shape-1-N'
  #train_split_strategy = 'combinatorial4-Y-4-2-X-4-2-Orientation-10-N-Scale-2-2-0FP_Shape-1-N'
  
  # Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  #train_split_strategy = 'combinatorial3-Y-4-2-X-4-2-Orientation-10-N-Scale-2-N-Shape-1-3'
  # Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  # --> the test and train set are not alternating sampling but rather completely distinct.
  train_split_strategy = 'combinatorial3-Y-4-S4-X-4-S4-Orientation-10-N-Scale-2-N-Shape-1-3'
  # Not too Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  # --> the test and train set are not alternating sampling but rather completely distinct.
  #train_split_strategy = 'combinatorial3-Y-2-S8-X-2-S8-Orientation-10-N-Scale-2-N-Shape-1-3'
  
  # Very Aggressively Undensified:
  #train_split_strategy = 'combinatorial3-Y-8-4-X-8-4-Orientation-10-N-Scale-2-N-1FP_Shape-1-N'
    

  test_split_strategy = train_split_strategy
  # Not too Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  # --> the test and train set are not alternating sampling but rather completely distinct.
  #test_split_strategy = 'combinatorial3-Y-2-S8-X-2-S8-Orientation-10-N-Scale-2-N-Shape-1-3'
  '''
  The issue with a train and test split with different density level is that some test values 
  on some latent axises may not appear in the train set (with different combinations than that
  of the test set), and so the system cannot get familiar to it... It is becomes a benchmark
  for both zero-shot composition learning and zero-shot components embedding (which could be
  a needed task in terms of analogy making: being able to understand that each latent axis
  can have unfamiliar values, i.e. associate the new values to the familiar latent axises...)
  '''
  # # Dataset:

  root = './datasets/dsprites-dataset'
  train_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=True, transform=rg_config['train_transform'], split_strategy=train_split_strategy)
  test_dataset = ReferentialGym.datasets.dSpritesDataset(root=root, train=False, transform=rg_config['test_transform'], split_strategy=test_split_strategy)

  train_i = 0
  test_i = 0
  continuer = True
  while continuer :
    train_i = train_i % len(train_dataset)
    test_i = test_i % len(test_dataset)

    train_sample = train_dataset[train_i]
    test_sample = test_dataset[test_i]

    train_img = train_sample[0].numpy().transpose( (1,2,0))
    print(train_img.shape)
    print('Sample latents :',train_sample[2])
    
    cv2.imshow('train', train_img )
    
    test_img = test_sample[0].numpy().transpose( (1,2,0))
    print(test_img.shape)
    print('Sample latents :',test_sample[2])
    
    cv2.imshow('test', test_img )
    
    while True :
      key = cv2.waitKey()
      if key == ord('q'):
        continuer = False
        break
      if key == ord('n') :
        train_i += 1
        break
      if key == ord('p') :
        train_i -= 1
        break
      if key == ord('i') :
        test_i += 1
        break
      if key == ord('j') :
        test_i -= 1
        break


if __name__ == '__main__':
    main()
