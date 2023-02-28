from typing import Dict, List, Tuple

from ReferentialGym.datasets import Dataset

import torch
from torch.utils.data import Dataset 

import os
import numpy as np
import copy
import random
from PIL import Image 
import h5py
from tqdm import tqdm


class DemonstrationDataset(Dataset) :
    def __init__(
        self, 
        replay_storage, 
        train=True, 
        transform=None, 
        split_strategy=None, 
        dataset_length=None,
        exp_key:str='succ_s',
    ) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.train = train
        self.replay_storage = replay_storage
        self.transform = transform
        self.split_strategy = split_strategy
        self.dataset_length = dataset_length

        # Load dataset
        print('Keys in the replay storage:')
        for k in self.replay_storage.keys: print(k)
        
        self.exp_key = exp_key
        
        self.action_set = set([a.item() for a in getattr(self.replay_storage, 'a')[0] if isinstance(a, torch.Tensor)])
        #self.reward_set = set(getattr(self.replay_storage, 'r'))
        
        self.latents_classes = np.zeros((len(self.replay_storage), 3))
        self.latents_values = np.zeros((len(self.replay_storage), 3))
        for idx in range(len(self.replay_storage)):
            action_idx = getattr(self.replay_storage, 'a')[0][idx].item()
            non_terminal_bool = getattr(self.replay_storage, 'non_terminal')[0][idx].item()
            
            reward_sign = 0
            reward = getattr(self.replay_storage, 'r')[0][idx].item() 
            if reward > 0:
                reward_sign = 2
            elif reward < 0:
                reward_sign = 1
            self.latents_classes[idx][0] = action_idx
            self.latents_classes[idx][1] = reward_sign
            self.latents_classes[idx][2] = non_terminal_bool
            
            self.latents_values[idx][0] = action_idx
            self.latents_values[idx][1] = reward
            self.latents_classes[idx][2] = non_terminal_bool

        self.latents_classes = self.latents_classes.astype(int)

        self.test_latents_mask = np.zeros_like(self.latents_classes)
        self.targets = np.zeros(len(self.latents_classes)) 
        
        lpd2tensor_mult = np.asarray([
            #3*len(self.action_set)*2,
            3*len(self.action_set),
            3,
            1]
        )

        for idx, latent_cls in enumerate(self.latents_classes):
            """
            self.targets[idx] = idx  
            """
            target = copy.deepcopy(latent_cls)
            target = target*lpd2tensor_mult
            self.targets[idx] = np.sum(target).item()
            
        if self.split_strategy is not None:
            strategy = self.split_strategy.split('-')
            if 'divider' in self.split_strategy and 'offset' in self.split_strategy:
                self.divider = int(strategy[1])
                assert(self.divider>0)
                self.offset = int(strategy[-1])
                assert(self.offset>=0 and self.offset<self.divider)
            elif 'combinatorial' in self.split_strategy:
                self.counter_test_threshold = int(strategy[0][len('combinatorial'):])
                # (default: 2) Specifies the threshold on the number of latent dimensions
                # whose values match a test value. Below this threshold, samples are used in training.
                # A value of 1 implies a basic train/test split that tests generalization to out-of-distribution values.
                # A value of 2 implies a train/test split that tests generalization to out-of-distribution pairs of values...
                # It implies that test value are encountered but never when combined with another test value.
                # It is a way to test for binary compositional generalization from well known stand-alone test values.
                # A value of 3 tests for ternary compositional generalization from well-known:
                # - stand-alone test values, and
                # - binary compositions of test values.
                
                '''
                With regards to designing axises as primitives:
                
                It implies that all the values on this latent axis are treated as test values
                when combined with a test value on any other latent axis.
                
                N.B.: it is not possible to test for out-of-distribution values in that context...
                N.B.1: It is required that the number of primitive latent axis be one less than
                        the counter_test_thershold, at most.

                A number of fillers along this primitive latent axis can then be specified in front
                of the FP pattern...
                Among the effective indices, those with an ordinal lower or equal to the number of
                filler allowed will be part of the training set.
                '''
                self.latent_dims = {}
                # self.strategy[0] : 'combinatorial'
                # 1: action_idx
                self.latent_dims['action_idx'] = {'size': len(self.action_set)}
                
                self.latent_dims['action_idx']['nbr_fillers'] = 0
                self.latent_dims['action_idx']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['action_idx']['primitive']:
                    self.latent_dims['action_idx']['nbr_fillers'] = int(strategy[1].split('FP')[0])

                self.latent_dims['action_idx']['position'] = 0
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[2]:
                    strategy[2] = strategy[2].split('RemainderToUse')
                    self.latent_dims['action_idx']['remainder_use'] = int(strategy[2][1])
                    strategy[2] = strategy[2][0]
                else:
                    self.latent_dims['action_idx']['remainder_use'] = 0
                self.latent_dims['action_idx']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['action_idx']['untested'] = True
                    self.latent_dims['action_idx']['test_set_divider'] = (self.latent_dims['action_idx']['size']//self.latent_dims['action_idx']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['action_idx']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['action_idx']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['action_idx']['test_set_divider'] = int(strategy[3])

                # 4: reward_sign
                self.latent_dims['reward_sign'] = {'size': 3}
                
                self.latent_dims['reward_sign']['nbr_fillers'] = 0
                self.latent_dims['reward_sign']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['reward_sign']['primitive']:
                    self.latent_dims['reward_sign']['nbr_fillers'] = int(strategy[4].split('FP')[0])

                self.latent_dims['reward_sign']['position'] = 1
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[5]:
                    strategy[5] = strategy[5].split('RemainderToUse')
                    self.latent_dims['reward_sign']['remainder_use'] = int(strategy[5][1])
                    strategy[5] = strategy[5][0]
                else:
                    self.latent_dims['reward_sign']['remainder_use'] = 0
                self.latent_dims['reward_sign']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['reward_sign']['untested'] = True
                    self.latent_dims['reward_sign']['test_set_divider'] = (self.latent_dims['reward_sign']['size']//self.latent_dims['reward_sign']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['reward_sign']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['reward_sign']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['reward_sign']['test_set_divider'] = int(strategy[6])
                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)

        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        self.traintest_indices = []
        if self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(len(self.replay_storage)):
                if idx % self.divider == self.offset:
                    self.indices.append(idx)

            self.train_ratio = 0.8
            end = int(len(self.indices)*self.train_ratio)

            self.traintest_indices = copy.deepcopy(self.indices)
            if self.train:
                self.indices = self.indices[:end]
            else:
                self.indices = self.indices[end:]

            print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
            print(f"Dataset Size: {len(self.indices)} out of {len(self.replay_storage)} : {100*len(self.indices)/len(self.replay_storage)}%.")
        elif 'combinatorial' in self.split_strategy:
            indices_latents = list(zip(range(self.latents_classes.shape[0]), self.latents_classes))
            for idx, latent_class in tqdm(indices_latents):
                effective_test_threshold = self.counter_test_threshold
                counter_test = {}
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[dim_dict['position']]
                    quotient = (dim_class+1)//dim_dict['divider']
                    remainder = (dim_class+1)%dim_dict['divider']
                    if remainder!=dim_dict['remainder_use']:
                        skip_it = True
                        break

                    if dim_dict['primitive']:
                        ordinal = quotient
                        if ordinal > dim_dict['nbr_fillers']:
                            effective_test_threshold -= 1

                    if 'test_set_divider' in dim_dict and quotient%dim_dict['test_set_divider']==0:
                        counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_end' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient > max_quotient-dim_dict['test_set_size_sample_from_end']:
                            counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_start' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient <= dim_dict['test_set_size_sample_from_start']:
                            counter_test[dim_name] = 1

                    if dim_name in counter_test:
                        self.test_latents_mask[idx, dim_dict['position']] = 1
                        
                if skip_it: continue

                self.traintest_indices.append(idx)
                if self.train:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        continue
                    else:
                        self.indices.append(len(self.traintest_indices)-1)
                        #self.indices.append(idx)
                else:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        self.indices.append(len(self.traintest_indices)-1)
                        #self.indices.append(idx)
                    else:
                        continue

            print(f"Split Strategy: {self.split_strategy}")
            print(self.latent_dims)
            print(f"Dataset Size: {len(self.indices)} out of {len(self.replay_storage)} : {100*len(self.indices)/len(self.replay_storage)}%.")
            
        max_values_per_latent = max(3, max(self.action_set)+1)
        self.latents_one_hot_encodings = np.eye(max_values_per_latent)[self.latents_classes.reshape(-1)]
        self.latents_one_hot_encodings = self.latents_one_hot_encodings.reshape((-1, self.latents_classes.shape[-1]*max_values_per_latent))
        
        """
        self.imgs = self.imgs[self.indices]
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        
        self.test_latents_mask = self.test_latents_mask[self.indices]
        self.targets = self.targets[self.indices]
        """

        #self.imgs = self.imgs[self.traintest_indices]
        self.latents_values = self.latents_values[self.traintest_indices]
        self.latents_classes = self.latents_classes[self.traintest_indices]
        self.latents_one_hot_encodings = self.latents_one_hot_encodings[self.traintest_indices]
        
        self.test_latents_mask = self.test_latents_mask[self.traintest_indices]
        self.targets = self.targets[self.traintest_indices]
        
        self.latents_classes_2_idx = { 
            tuple(lc.tolist()): idx 
            for idx,lc in enumerate(self.latents_classes)
        }

        self.nbr_attributes_per_latent_dimension = {}
        for attr_id in range(self.latents_classes.shape[1]):
            values = set(self.latents_classes[:,attr_id]) 
            self.nbr_attributes_per_latent_dimension[attr_id] = {
                'size': len(values),
                'values': list(values),
            }
        
        print("Dataset : nbr of attributes per latent:", self.nbr_attributes_per_latent_dimension)

        print('Dataset loaded : OK.')
    
    def get_imgs(self, indices, key='s'):
        if isinstance(indices, int):    indices = [indices]
        indices_ = []
        for idx in indices:
            indices_.append(self.traintest_indices[idx])
        return getattr(self.replay_storage, key)[0][indices_]

    def sample_factors(self, num, random_state=None):
        """
        Sample a batch of factors Y.
        """
        if random_state is not None:
            factors_indices = random_state.choice(list(range(len(self.traintest_indices))), size=(num,), replace=True)
        else:
            factors_indices = np.random.choice(list(range(len(self.traintest_indices))), size=(num,), replace=True)
        
        factors = np.stack(self.latents_classes[factors_indices], axis=0)

        return factors
    
    def sample_latents_values_from_factors(self, factors, random_state=None):
        """
        Sample a batch of latents_values X given a batch of factors Y.
        """
        self.factors_indices = [] 
        
        for factor in factors:
            self.factors_indices.append(self.latents_classes_2_idx[tuple(factor.tolist())])
        
        #self.factors_indices = self.traintest_indices[self.factors_indices]
        latents_values = [lv for lv in self.latents_values[self.factors_indices]]
        
        return latents_values

    def sample_latents_ohe_from_factors(self, factors, random_state=None):
        """
        Sample a batch of latents_values X given a batch of factors Y.
        """
        self.factors_indices = [] 
        
        for factor in factors:
            self.factors_indices.append(self.latents_classes_2_idx[tuple(factor.tolist())])

        #self.factors_indices = self.traintest_indices[self.factors_indices]
        latents_ohe = [lohe for lohe in self.latents_one_hot_encodings[self.factors_indices]]
        
        return latents_ohe
        
    def sample_observations_from_factors(self, factors, random_state=None):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        self.factors_indices = [] 
        
        for factor in factors:
            self.factors_indices.append(self.latents_classes_2_idx[tuple(factor.tolist())])

        #self.factors_indices = self.traintest_indices[self.factors_indices]
        images = [
            im #Image.fromarray(im, mode='RGB') 
            for im in self.get_imgs(
                indices=self.factors_indices,
                key=self.exp_key,
            )
        ]
        #images = [Image.fromarray((im*255).astype('uint8')) for im in self.imgs[self.factors_indices]]
        
        if self.transform is not None:
            images = [self.transform(im) for im in images]
        
        images = torch.stack(images, dim=0)
        
        return images

    def __len__(self) -> int:
        if self.dataset_length is not None:
            return self.dataset_length
        return len(self.indices)

    def getclass(self, idx):
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        trueidx = self.indices[idx]
        target = self.targets[trueidx]
        return target

    def getlatentvalue(self, idx):
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        trueidx = self.indices[idx]
        latent_value = self.latents_values[trueidx]
        return latent_value

    def getlatentclass(self, idx):
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        trueidx = self.indices[idx]
        latent_class = self.latents_classes[trueidx]
        return latent_class

    def getlatentonehot(self, idx):
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        trueidx = self.indices[idx]
        latent_one_hot = self.latents_one_hot_encodings[trueidx]
        return latent_one_hot

    def gettestlatentmask(self, idx):
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        trueidx = self.indices[idx]
        test_latents_mask = self.test_latents_mask[trueidx]
        return test_latents_mask

    def __getitem__(self, idx:int) -> Dict[str,torch.Tensor]:
        """
        :param idx: Integer index.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representation of the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representation of the latent of the experience in value form.
                - `"exp_latents_one_hot_encoded"`: Tensor representation of the latent of the experience in one-hot-encoded class form.
                - `"exp_test_latent_mask"`: Tensor that highlights the presence of test values, if any on each latent axis.
        """
        if idx >= len(self.indices):
            idx = idx%len(self.indices)
        
        orig_idx = idx
        trueidx = self.indices[idx]

        #image = Image.fromarray((self.imgs[trueidx]*255).astype('uint8'))
        #trueidx = self.traintest_indices[trueidx]
        #image = Image.fromarray(self.get_imgs(indices=trueidx, key=self.exp_key), mode='RGB')
        image = self.get_imgs(indices=trueidx, key=self.exp_key)
        image = image[0][0]
        #image.requires_grad=True
        
        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_class = torch.from_numpy(self.getlatentclass(idx))
        latent_one_hot_encoded = torch.from_numpy(self.getlatentonehot(idx))
        test_latents_mask = torch.from_numpy(self.gettestlatentmask(idx))

        if self.transform is not None:
            image = self.transform(image)
        
        sampled_d = {
            "experiences":image, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value,
            "exp_latents_one_hot_encoded":latent_one_hot_encoded,
            "exp_test_latents_masks":test_latents_mask,
        }

        return sampled_d
