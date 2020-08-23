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

  egocentric = True 

  # In[23]:
  nbr_epoch = 100
  cnn_feature_size = 512 # 128 512 #1024
  stimulus_resize_dim = 64 #32
  normalize_rgb_values = False 
  rgb_scaler = 1.0 #255.0
  from ReferentialGym.datasets.utils import ResizeNormalize
  transform = ResizeNormalize(size=stimulus_resize_dim, 
                              normalize_rgb_values=normalize_rgb_values,
                              rgb_scaler=rgb_scaler)

  transform_degrees = 6
  transform_translate = (0.0625, 0.0625)

  from ReferentialGym.datasets.utils import AddEgocentricInvariance
  ego_inv_transform = AddEgocentricInvariance()

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

  if egocentric:
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

  # # Dataset:

  '''
  generate = True 
  img_size = 64
  nb_objects = 3
  nb_shapes = 3

  fontScale=0.5
  thickness=2
  '''

  generate = False
  random_generation = True
  
  """
  img_size = 64
  nb_objects = 3 #2
  nb_shapes = 4

  train_nb_rhs = 2
  """
  img_size = 128 #64
  nb_objects = 5 #2
  nb_shapes = 36

  train_nb_rhs = 2

  # If divided by 2, then not enough positions...
  #thickness=2
  #fontScale=0.5
  fontScale=1.0 #0.5
  thickness=2#1
  
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
  train_split_strategy = None #'uniformBinaryRelationalQuery' 
  nb_samples = 512 #int(1e3)
  
  # Normally dense: 
  #train_split_strategy = 'combinatorial1-Y-1-N-X-1-N-2IWP_Shape-1-N' 
  # Sparser: / (2*2)^NB_OBJECTS 
  #train_split_strategy = 'combinatorial1-Y-2-N-X-2-N-2IWP_Shape-1-N' 
  
  # Experiment 2: mistake?
  #train_split_strategy = 'combinatorial3-Y-4-E4-X-4-S4-Orientation-10-N-Scale-2-N-Shape-1-3'
  # Experiment 2: correct one? Most likely
  #train_split_strategy = 'combinatorial3-Y-4-S4-X-4-S4-Orientation-10-N-Scale-2-N-Shape-1-3'
  # Not too Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  # --> the test and train set are not alternating sampling but rather completely distinct.
  #train_split_strategy = 'combinatorial3-Y-2-S8-X-2-S8-Orientation-10-N-Scale-2-N-Shape-1-3'
  

  ## Test set:

  # Experiment 1:
  test_split_strategy = train_split_strategy
  

  root = './datasets/SQOOT-dataset'
  if random_generation:
    root += f'RandomGeneration-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}-size{nb_samples}'
  else:
    root += f'-imgS{img_size}-obj{nb_objects}-shapes{nb_shapes}-fS{fontScale}-th{thickness}'
  
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
    thickness=thickness)
  
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
    thickness=thickness)

  test_dataset.training_rhs = train_dataset.training_rhs
  test_dataset.testing_rhs = train_dataset.testing_rhs

  train_i = 0
  test_i = 0
  continuer = True
  while continuer :
    train_i = train_i % len(train_dataset)
    test_i = test_i % len(test_dataset)

    train_sample = train_dataset[train_i]
    test_sample = test_dataset[test_i]

    train_img = train_sample["experiences"].numpy().transpose( (1,2,0))
    print(train_img.shape)
    print('Sample latents :',train_sample["exp_latents"])
    
    cv2.imshow('train', train_img )
    
    test_img = test_sample["experiences"].numpy().transpose( (1,2,0))
    print(test_img.shape)
    print('Sample latents :',test_sample["exp_latents"])
    
    cv2.imshow('test', test_img )
    
    while True :
      key = cv2.waitKey()
      if key == ord('q'):
        continuer = False
        #import ipdb; ipdb.set_trace()
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
