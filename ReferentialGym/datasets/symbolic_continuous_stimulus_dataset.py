from typing import Dict, List

import sys
import random
import numpy as np 
import argparse 
import copy

import torch
from torch.utils.data import Dataset 
from PIL import Image 

eps = 1e-8


class SymbolicContinuousStimulusDataset(Dataset) :
    def __init__(
        self, 
        train=True, 
        transform=None, 
        split_strategy=None, 
        nbr_latents=10, 
        min_nbr_values_per_latent=2, 
        max_nbr_values_per_latent=10, 
        nbr_object_centric_samples=1,
        dataset_length=None,
        prototype=None) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.nbr_latents = nbr_latents
        self.min_nbr_values_per_latent = min_nbr_values_per_latent
        self.max_nbr_values_per_latent = max_nbr_values_per_latent
        self.nbr_object_centric_samples = nbr_object_centric_samples
        
        self.test_set_divider = 2

        self.prototype = prototype
        self.dataset_length = dataset_length

        self.reset()
        self.imgs = [np.zeros((64,64,3))]

        self.train = train
        self.transform = transform
        self.split_strategy = split_strategy

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

                nbr_primitives_and_tested = len([
                    k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]
                ])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)
            else:
                raise NotImplementedError
        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        self.traintest_indices = []
        if self.prototype is not None:
            assert not(self.train)
            self.indices = [idx for idx in range(self.dataset_size) if idx not in self.prototype.indices]
            print(f"Split Strategy: {self.split_strategy}")
            print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size} : {100*len(self.indices)/self.dataset_size}%.")

        elif self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(self.dataset_size):
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
            print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size}: {100*len(self.indices)/self.dataset_size}%.")
        elif 'combinatorial' in self.split_strategy:
            self.traintest_indices = []
            for idx in range(self.dataset_size):
                object_centric_sidx = idx//self.nbr_object_centric_samples
                coord = self.idx2coord(object_centric_sidx)
                latent_class = np.array(coord)
                
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
                        self.indices.append(idx)
                else:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        self.indices.append(idx)
                    else:
                        continue

            print(f"Split Strategy: {self.split_strategy}")
            print(f"Dataset Size: {len(self.indices)} out of {self.dataset_size} : {100*len(self.indices)/self.dataset_size}%.")
        else:
            raise NotImplementedError            

        self.traintest_targets = self.targets[self.traintest_indices]
        self.targets = self.targets[self.indices]
        
        self.latents_classes = []
        for trueidx in self.traintest_indices:
            object_centric_sidx = trueidx//self.nbr_object_centric_samples
            coord = self.idx2coord(object_centric_sidx)
            latent_class = np.array(coord)
            self.latents_classes.append(latent_class)
        self.latents_classes = np.asarray(self.latents_classes)
        print('Dataset loaded : OK.')
    
    def reset(self):
        global eps 

        if self.prototype is None:
            self.latent_dims = {}
            self.latent_sizes = []
            self.dataset_size = 1
            for l_idx in range(self.nbr_latents):
                l_size = np.random.randint(low=self.min_nbr_values_per_latent, high=self.max_nbr_values_per_latent+1)
                self.dataset_size *= l_size
                self.latent_sizes.append(l_size)
                self.latent_dims[l_idx] = {'size': l_size}
                
                self.latent_dims[l_idx]['value_section_size'] = 2.0/l_size
                self.latent_dims[l_idx]['max_sigma'] = self.latent_dims[l_idx]['value_section_size']/6
                self.latent_dims[l_idx]['min_sigma'] = self.latent_dims[l_idx]['value_section_size']/12
                self.latent_dims[l_idx]['sections'] = {}
                for s_idx in range(l_size):
                    s_d = {}
                    s_d['section_offset'] = -1+s_idx*self.latent_dims[l_idx]['value_section_size']
                    s_d['sigma'] = np.random.uniform(
                        low=self.latent_dims[l_idx]['min_sigma']+eps,
                        high=self.latent_dims[l_idx]['max_sigma']-eps,
                    )
                    s_d['safe_section_size'] = self.latent_dims[l_idx]['value_section_size'] - 6*s_d['sigma']
                    s_d['safe_section_mean_offset'] = 3*s_d['sigma']
                    s_d['mean_ratio'] = np.random.uniform(low=0,high=1.0)
                    s_d['mean'] = s_d['section_offset'] + s_d['safe_section_mean_offset'] + s_d['mean_ratio'] * s_d['safe_section_size']
                    
                    self.latent_dims[l_idx]['sections'][s_idx] = s_d

                self.latent_dims[l_idx]['nbr_fillers'] = 0
                self.latent_dims[l_idx]['primitive'] = False
                self.latent_dims[l_idx]['position'] = l_idx
                self.latent_dims[l_idx]['remainder_use'] = 0
                self.latent_dims[l_idx]['divider'] = 1 # no need to divide as it is fully parameterized
                self.latent_dims[l_idx]['test_set_divider'] = self.test_set_divider

            self.dataset_size *= self.nbr_object_centric_samples
            self.generate_object_centric_samples()

            self.latent_strides = [1]
            dims = [ld['size'] for ld in self.latent_dims.values()]
            for idx in range(self.nbr_latents):
                self.latent_strides.append(np.prod(dims[-idx-1:]))
            self.latent_strides = list(reversed(self.latent_strides[:-1]))
            
            self.test_latents_mask = np.zeros((self.dataset_size, self.nbr_latents))
        else:
            self.latent_dims = self.prototype.latent_dims
            self.latent_sizes = self.prototype.latent_sizes
            self.dataset_size = self.prototype.dataset_size
            self.latent_strides = self.prototype.latent_strides
            self.test_latents_mask = self.prototype.test_latents_mask

        self.targets = np.zeros(self.dataset_size)
        for idx in range(self.dataset_size):
            self.targets[idx] = idx//self.nbr_object_centric_samples

    def generate_object_centric_samples(self):
        """
        """
        for lidx in range(self.nbr_latents):
            for lvalue in range(self.latent_dims[lidx]['size']):
                oc_samples = []
                for oc_sidx in range(self.nbr_object_centric_samples):
                    lvalue_sample = np.random.normal(
                        loc=self.latent_dims[lidx]['sections'][lvalue]['mean'],
                        scale=self.latent_dims[lidx]['sections'][lvalue]['sigma'],
                    )
                    oc_samples.append(lvalue_sample)
                self.latent_dims[lidx]['sections'][lvalue]['object_centric_samples'] = oc_samples

    def generate_object_centric_observations(self, latent_class, sample=True):
        """
        :arg latent_class: Numpy.ndarray of shape (batch_size, self.nbr_latents).

        :return observations: Numpy.ndarray of shape (batch_size, self.nbr_latents) with
            values on each dimension sampled from the corresponding value's (gaussian) 
            distribution.
        """
        batch_size = latent_class.shape[0]
        object_centric_sample_idx = np.random.randint(low=0,high=self.nbr_object_centric_samples)

        observations = np.zeros((batch_size, self.nbr_latents))
        for bidx in range(batch_size):
            for lidx in range(self.nbr_latents):
                lvalue = latent_class[bidx,lidx]
                """
                lvalue_sample = np.random.choice(
                    a=self.latent_dims[lidx]['sections'][lvalue]['object_centric_samples'],
                    size=1,
                )
                """
                if sample:
                    lvalue_sample = self.latent_dims[lidx]['sections'][lvalue]['object_centric_samples'][object_centric_sample_idx]
                else:
                    lvalue_sample = self.latent_dims[lidx]['sections'][lvalue]['mean']
                observations[bidx,lidx] = float(lvalue_sample)

        return observations

  
    def generate_observations(self, latent_class, sample=True):
        """
        :arg latent_class: Numpy.ndarray of shape (batch_size, self.nbr_latents).
        :arg sample: Bool, if `True`, then values are sampled from each distribution.
            Otherwise, the mean value is used.
        :return observations: Numpy.ndarray of shape (batch_size, self.nbr_latents) with
            values on each dimension sampled from the corresponding value's (gaussian) 
            distribution.
        """
        batch_size = latent_class.shape[0]

        observations = np.zeros((batch_size, self.nbr_latents))
        for bidx in range(batch_size):
            for lidx in range(self.nbr_latents):
                lvalue = latent_class[bidx,lidx]
                if sample:
                    lvalue_sample = np.random.normal(
                        loc=self.latent_dims[lidx]['sections'][lvalue]['mean'],
                        scale=self.latent_dims[lidx]['sections'][lvalue]['sigma'],
                    )
                else:
                    lvalue_sample = self.latent_dims[lidx]['sections'][lvalue]['mean']
                observations[bidx,lidx] = lvalue_sample

        return observations

    def coord2idx(self, coord):
        """
        WARNING: the object-centrism is not taken into account here.

        :arg coord: List of self.nbr_latents elements.        
        
        :return idx: Int, corresponding index.
        """
        idx = 0
        for stride, mult in zip(self.latent_strides,coord):
            idx += stride*mult
        return idx

    def idx2coord(self, idx):
        """
        WARNING: the object-centrism MUST be taking into account
        before calling this function.

        :arg idx: Int, must be contained within [0, self.dataset_size/self.nbr_object_centric_samples].

        :return coord: List of self.nbr_latents elements corresponding the entry of :arg idx:.
        """
        coord = []
        remainder = idx
        for lidx in range(self.nbr_latents):
            coord.append(remainder//self.latent_strides[lidx])
            remainder = remainder % self.latent_strides[lidx]
        return coord 
    
    def __len__(self) -> int:
        if self.dataset_length is not None:
            return self.dataset_length
        return len(self.indices)
    

    def getclass(self, idx, from_traintest=False):
        if from_traintest:
            indices = self.traintest_indices
        else:
            indices = self.indices
        if idx >= len(indices):
            idx = idx%len(indices)
        if from_traintest:
            target = self.traintest_targets[idx]
        else:
            target = self.targets[idx]
        return target

    def getlatentvalue(self, idx, from_traintest=False):
        if from_traintest:
            indices = self.traintest_indices
        else:
            indices = self.indices
        if idx >= len(indices):
            idx = idx%len(indices)
        trueidx = indices[idx]
        object_centric_sidx = trueidx//self.nbr_object_centric_samples
        coord = self.idx2coord(object_centric_sidx)
        latent_class = np.array(coord).reshape((1,-1))
        latent_value = self.generate_observations(latent_class, sample=False)
        return latent_value

    def getlatentclass(self, idx, from_traintest=False):
        if from_traintest:
            indices = self.traintest_indices
        else:
            indices = self.indices
        if idx >= len(indices):
            idx = idx%len(indices)
        trueidx = indices[idx]
        object_centric_sidx = trueidx//self.nbr_object_centric_samples
        coord = self.idx2coord(object_centric_sidx)
        latent_class = np.array(coord)
        return latent_class

    def getlatentonehot(self, idx, from_traintest=False):
        if from_traintest:
            indices = self.traintest_indices
        else:
            indices = self.indices
        if idx >= len(indices):
            idx = idx%len(indices)
        # object-centrism is taken into account in getlatentclass fn:
        latent_class = self.getlatentclass(idx, from_traintest=from_traintest)
        latent_one_hot_encoded = np.zeros(sum(self.latent_sizes))
        startidx = 0
        for lsize, lvalue in zip(self.latent_sizes, latent_class):
            latent_one_hot_encoded[startidx+lvalue] = 1
            startidx += lsize 
        return latent_one_hot_encoded

    def gettestlatentmask(self, idx, from_traintest=False):
        if from_traintest:
            indices = self.traintest_indices
        else:
            indices = self.indices
        if idx >= len(indices):
            idx = idx%len(indices)
        trueidx = indices[idx]
        test_latents_mask = self.test_latents_mask[trueidx]
        return test_latents_mask

    def sample_factors(self, num, random_state=None):
        """
        Sample a batch of factors Y.
        """
        #return random_state.randint(low=0, high=self.nbr_values_per_latent, size=(num, self.nbr_latents))
        # It turns out the random state is not really being updated apparently.
        # Therefore it was always sampling the same values...
        random_indices = np.random.randint(low=0, high=self.dataset_size, size=(num,))
        return np.stack([self.getlatentclass(ridx, from_traintest=True) for ridx in random_indices], axis=0)
        
    def sample_observations_from_factors(self, factors, random_state=None):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        obs = torch.from_numpy(
            self.generate_object_centric_observations(factors),
        )
        return obs.float()

    def sample_latents_values_from_factors(self, factors, random_state=None):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        return self.generate_observations(factors, sample=False)

    def sample_latents_ohe_from_factors(self, factors, random_state=None):
        """
        Sample a batch of observations X given a batch of factors Y.
        """
        batch_size = factors.shape[0]
        ohe = np.zeros((batch_size, sum(self.latent_sizes)))
        for bidx in range(batch_size):
            idx = self.coord2idx(factors[bidx])
            ohe[bidx] = self.getlatentonehot(idx)
        return ohe

    def sample(self, num, random_state=None):
        """
        Sample a batch of factors Y and observations X.
        """
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

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

        latent_class = self.getlatentclass(idx)
        stimulus = torch.from_numpy(
                self.generate_object_centric_observations(
                    latent_class.reshape(
                        (1,-1)
                    )
                )
        ).float()
        latent_class = torch.from_numpy(latent_class)

        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_one_hot_encoded = torch.from_numpy(self.getlatentonehot(idx))
        test_latents_mask = torch.from_numpy(self.gettestlatentmask(idx))

        if self.transform is not None:
            stimulus = self.transform(stimulus)
        
        sampled_d = {
            "experiences":stimulus, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value,
            "exp_latents_one_hot_encoded":latent_one_hot_encoded,
            "exp_test_latents_masks":test_latents_mask,
        }

        return sampled_d
