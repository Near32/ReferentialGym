import random
import numpy as np
import cv2 

import ReferentialGym

import torch
import torchvision
import torchvision.transforms as T 


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE

def tsne(data, nbr_train):
  fig = plt.figure()
  emb = TSNE(
    n_components=2,
    perplexity=5,
    init='pca',
    verbose=2,
    random_state=7,
    learning_rate=300,
    n_iter=10000).fit_transform(data)

  train = plt.scatter(emb[:nbr_train,0], emb[:nbr_train,1], c='skyblue', s=50, alpha=0.5)
  test = plt.scatter(emb[nbr_train:,0], emb[nbr_train:,1], c='red', s=5, alpha=0.5)
  
  plt.legend((train, test),
           ('Train', 'Test'),
           #scatterpoints=1,
           #loc='lower left',
           #ncol=3,
           fontsize=8)

  plt.show()
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  emb3d = TSNE(
    n_components=3,
    perplexity=500,
    init='pca',
    verbose=1,
    random_state=7).fit_transform(data)
  train = ax.scatter(emb3d[:nbr_train,0], emb3d[:nbr_train,1], emb3d[:nbr_train,2], c='skyblue', s=50)
  test = ax.scatter(emb3d[nbr_train:,0], emb3d[nbr_train:,1], emb3d[nbr_train:,2], c='red', s=5)
  
  plt.show()
  '''

def plot(train_dataset, test_dataset):
  #dd = {'label':[]}
  dd = {}

  for idx in range(len(train_dataset)):
    latent_repr = train_dataset.getlatentclass(idx)
    for key in train_dataset.latent_dims.keys():
      pos = train_dataset.latent_dims[key]['position']
      if key not in dd: 
        dd[key] = [] 
      dd[key].append(latent_repr[pos])
    #dd['label'].append('train')
  nbr_train = idx+1 
  
  for idx in range(len(test_dataset)):
    latent_repr = test_dataset.getlatentclass(idx)
    for key in test_dataset.latent_dims.keys():
      pos = test_dataset.latent_dims[key]['position']
      if key not in dd: dd[key] = [] 
      dd[key].append(latent_repr[pos])
    #dd['label'].append('test')

  df = pd.DataFrame(dd)
  #df = pd.DataFrame({'X':range(1,101), 'Y':np.random.rand(100)*15+range(1,101), 'Z':(np.random.randn(100)*15+range(1,101))*2})

  tsne(df.to_numpy(), nbr_train)


  variations = {}
  for key in dd.keys():
    variations[key] = len(np.unique(np.asarray(dd[key])))

  nbr_lines = variations['Shape']
  nbr_rows = variations['Scale']

  
  fig = plt.figure()
  axes = []
  idxcurrentgraph = 0
  for line in range(nbr_lines):
    for row in range(nbr_rows):
      idxcurrentgraph += 1
      ax = fig.add_subplot(nbr_lines*100+nbr_rows*10+idxcurrentgraph, projection='3d')
      ax.scatter(df['X'][:nbr_train], df['Y'][:nbr_train], df['Orientation'][:nbr_train], c='skyblue', s=50)
      ax.scatter(df['X'][nbr_train:], df['Y'][nbr_train:], df['Orientation'][nbr_train:], c='red', s=5)
      #ax.view_init(125, 35)
      #ax.view_init(145, 45)
      #ax.view_init(145, -135)
      ax.view_init(-175, 145)
      ax.set_xlabel('position X')
      ax.set_ylabel('position Y')
      ax.set_zlabel('Orientaiton')
      axes.append(ax)
  plt.show()

  return



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

  from ReferentialGym.datasets.utils import AddEgocentricInvariance
  ego_inv_transform = AddEgocentricInvariance()


  transform_degrees = 25
  transform_translate = (0.0625, 0.0625)

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
  
      "train_transform":            transform,
      "test_transform":             transform,
  }

  if True:
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
  
      
  # INTER SIMPLE X+Y
  # Sparse: simple splitted XY X 4/ Y 4/ --> 16 test / 48 train 
  #train_split_strategy = 'combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N' 
  # Dense: simple splitted XY X 8/ Y 8/ --> 64 test / 192 train 
  #train_split_strategy = 'combinatorial2-Y-2-2-X-2-2-Orientation-40-N-Scale-6-N-Shape-3-N' 
  # Denser: simple splitted XY X 16/ Y 16/ --> 256 test / 768 train 
  #train_split_strategy = 'combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N' 
  
  # EXTRA SIMPLE X+Y
  # Sparse: simple splitted XY X 4/ Y 4/ --> 16 test / 48 train 
  #train_split_strategy = 'combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N' 
  
  # INTER SIMPLE Scale+Orientation
  # Sparse: simple splitted Scale Orientation Scale 4/ Orientation 3/ --> 12 test / 36 train 
  #train_split_strategy = 'combinatorial2-Y-32-N-X-32-N-Orientation-5-2-Scale-1-2-Shape-3-N' 
  # Dense: simple splitted Scale Orientation Scale 10/ Orientation 3/ --> 30 test / 90 train 
  #train_split_strategy = 'combinatorial2-Y-32-N-X-32-N-Orientation-2-2-Scale-1-2-Shape-3-N' 
  
  # EXTRA SIMPLE Scale+Orientation
  # Sparse: simple splitted Scale Orientation Scale 4/ Orientation 3/ --> 12 test / 36 train 
  #train_split_strategy = 'combinatorial2-Y-32-N-X-32-N-Orientation-5-2-Scale-1-2-Shape-3-N' 
  # Dense: simple splitted Scale Orientation Scale 10/ Orientation 3/ --> 30 test / 90 train 
  #train_split_strategy = 'combinatorial2-Y-32-N-X-32-N-Orientation-2-S10-Scale-1-S3-Shape-3-N' 
  
  # INTER MULTI3 X+Y+Orientation
  #COMB2 Sparse: simple splitted XYOrientation X 4/ Y 4/ Orientation 4/ --> 256 test / 256 train 
  #train_split_strategy = 'combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N' 
  
  # EXTRA MULTI3 X+Y+Orientation
  #COMB2 Sparse: simple splitted XYOrientation X 4/ Y 4/ Orientation 4/ --> 256 test / 256 train 
  train_split_strategy = 'combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N' 
    

  # INTER MULTI:
  #COMB2: Sparser 4 Attributes: 264 test / 120 train
  #train_split_strategy = 'combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-1-2-Shape-3-N' 
  #COMB4: Sparser 4 Attributes: 24 test / 360 train
  #train_split_strategy = 'combinatorial4-Y-8-2-X-8-2-Orientation-10-2-Scale-1-2-Shape-3-N' 
  
  # COMB2: Sparse 4 Attributes: 2112 test / 960 train
  #train_split_strategy = 'combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-1-2-Shape-3-N' 
  


  #Multi 3: denser simple X+Y with the sample size of multi 4:
  #train_split_strategy = 'combinatorial2-Y-1-S16-X-1-S16-Orientation-2-S10-Scale-6-N-Shape-3-N' 
  

  #MULTI:
  #COMB2: Sparser 4 Attributes: 264 test / 120 train
  #train_split_strategy = 'combinatorial2-Y-8-S2-X-8-S2-Orientation-10-S2-Scale-1-S3-Shape-3-N' 
  #COMB4: Sparser 4 Attributes: 24 test / 360 train
  #train_split_strategy = 'combinatorial4-Y-8-S2-X-8-S2-Orientation-10-S2-Scale-1-S3-Shape-3-N' 
  
  #COMB2: Sparse 4 Attributes: 2112 test / 960 train
  #train_split_strategy = 'combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-1-S3-Shape-2-N' 
  #COMB2: Dense 4 Attributes: 21120 test / 9600 train
  #train_split_strategy = 'combinatorial2-Y-2-S8-X-2-S8-Orientation-2-S10-Scale-1-S3-Shape-2-N' 
             
  #COMB4: Sparse: multi 4 splitted XY X 4/ Y 4/ Orientation 4/ Scale 3/-->  192 test / 1344 train 
  #train_split_strategy = 'combinatorial4-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-1-S3-Shape-3-N' 
  #COMB4: Dense: multi 4 splitted XY X 8/ Y 8/ Orientation 8/ Scale 6/-->  1536 test / 10752 train 
  #train_split_strategy = 'combinatorial4-Y-2-S8-X-2-S8-Orientation-2-S10-Scale-1-S3-Shape-3-N' 
  
  # Dense: Multi X 8/ Y 8/ Orientation 10/ NOT Scale 1/ Shape 3
  #train_split_strategy = 'combinatorial3-Y-2-8-X-2-8-Orientation-2-10-Scale-6-N-Shape-1-N' 
  # Sparse: Multi X 4/ Y 4/ Orientation 4/ NOT Scale 1/ Shape 2
  #train_split_strategy = 'combinatorial2-Y-4-4-X-4-4-Orientation-5-4-Scale-6-N-Shape-1-2' 
  # Sparser: Multi X 2/ Y 2/ Orientation 2/ NOT Scale 1/ Shape 2
  #train_split_strategy = 'combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-6-N-Shape-1-1'
  # Even Sparser: Multi X 2/ Y 2/ Orientation 2/ NOT Scale 1/ NOT Shape 1
  #train_split_strategy = 'combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-6-N-Shape-3-N' 
  
  '''
  TEST Even Sparser: Multi X 2/ Y 2/ NOT Orientation 1/ NOT Scale 1/ NOT Shape 1
  The rule of thumb when it comes to selecting the number of testing axises is 
  to understand it as a threshold so that when there is more test-only values 
  taken then it will obviously push the stimulus in the test set.
  Thus, it reduces the train set size further compared to the test set size.
  '''

  #train_split_strategy = 'combinatorial2-Y-8-2-X-8-2-Orientation-10-2-Scale-6-N-Shape-3-N' 
  #train_split_strategy = 'combinatorial1-Y-8-2-X-8-2-Orientation-40-N-Scale-6-N-Shape-3-N' 
  
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
  # Not too Agressive: compositional extrapolation is tested on Heart Shape at 16 positions...
  # --> the test and train set are not alternating sampling but rather completely distinct.
  # Experiment 2: definitively splitted with extrapolation when assuming 4-s4
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

  if False: plot(train_dataset, test_dataset)

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
