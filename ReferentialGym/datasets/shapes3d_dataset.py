from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset 
import os
import numpy as np
import copy
import random
from PIL import Image 
import h5py
from tqdm import tqdm


class Shapes3DDataset(Dataset) :
    def __init__(self, root='./', train=True, transform=None, split_strategy=None, dataset_length=None) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.train = train
        self.root = os.path.join(root, '3dshapes.h5')
        self.transform = transform
        self.split_strategy = split_strategy
        self.dataset_length = dataset_length

        # Load dataset
        dataset_zip = h5py.File(self.root, 'r')
        #dataset_zip = np.load(self.root, encoding='latin1', allow_pickle=True)
        print('Keys in the dataset:')
        for k in dataset_zip.keys(): print(k)
        self.imgs = dataset_zip['images'][:]
        self.latents_values = np.array(dataset_zip['labels'][:])
        self.latents_classes = np.array(dataset_zip['labels'][:])
        self.latents_classes[:, 0:3] = (self.latents_classes[:,0:3].astype(float)*10).astype(int)
        self.latents_classes[:, 3] = ((self.latents_classes[:,3].astype(float)*100-7.5)/7).astype(int)-9
        self.latents_classes[:, 4] = self.latents_classes[:, 4].astype(int)
        self.latents_classes[:, 5] = ((self.latents_classes[:,5].clip(-30,29).astype(float)*1+30.0)/4).astype(int)
        self.latents_classes = self.latents_classes.astype(int)

        self.test_latents_mask = np.zeros_like(self.latents_classes)
        self.targets = np.zeros(len(self.latents_classes)) #[random.randint(0, 10) for _ in self.imgs]
        lpd2tensor_mult = np.asarray([
            15*4*8*10*10,
            15*4*8*10,
            15*4*8,
            15*4,
            15,
            1]
        )

        for idx, latent_cls in enumerate(self.latents_classes):
            """
            self.targets[idx] = idx  
            """
            target = copy.deepcopy(latent_cls)
            # nullify Orientation: 
            target[5] = 0
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
                # 1: floor_hue
                self.latent_dims['floor_hue'] = {'size': 10}
                
                self.latent_dims['floor_hue']['nbr_fillers'] = 0
                self.latent_dims['floor_hue']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['floor_hue']['primitive']:
                    self.latent_dims['floor_hue']['nbr_fillers'] = int(strategy[1].split('FP')[0])

                self.latent_dims['floor_hue']['position'] = 0
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[2]:
                    strategy[2] = strategy[2].split('RemainderToUse')
                    self.latent_dims['floor_hue']['remainder_use'] = int(strategy[2][1])
                    strategy[2] = strategy[2][0]
                else:
                    self.latent_dims['floor_hue']['remainder_use'] = 0
                self.latent_dims['floor_hue']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['floor_hue']['untested'] = True
                    self.latent_dims['floor_hue']['test_set_divider'] = (self.latent_dims['floor_hue']['size']//self.latent_dims['floor_hue']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['floor_hue']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['floor_hue']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['floor_hue']['test_set_divider'] = int(strategy[3])

                # 4: wall_hue
                self.latent_dims['wall_hue'] = {'size': 10}
                
                self.latent_dims['wall_hue']['nbr_fillers'] = 0
                self.latent_dims['wall_hue']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['wall_hue']['primitive']:
                    self.latent_dims['wall_hue']['nbr_fillers'] = int(strategy[4].split('FP')[0])

                self.latent_dims['wall_hue']['position'] = 1
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[5]:
                    strategy[5] = strategy[5].split('RemainderToUse')
                    self.latent_dims['wall_hue']['remainder_use'] = int(strategy[5][1])
                    strategy[5] = strategy[5][0]
                else:
                    self.latent_dims['wall_hue']['remainder_use'] = 0
                self.latent_dims['wall_hue']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['wall_hue']['untested'] = True
                    self.latent_dims['wall_hue']['test_set_divider'] = (self.latent_dims['wall_hue']['size']//self.latent_dims['wall_hue']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['wall_hue']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['wall_hue']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['wall_hue']['test_set_divider'] = int(strategy[6])
                # 7: object_hue
                self.latent_dims['object_hue'] = {'size': 10}
                
                self.latent_dims['object_hue']['nbr_fillers'] = 0
                self.latent_dims['object_hue']['primitive'] = ('FP' in strategy[7])
                if self.latent_dims['object_hue']['primitive']:
                    self.latent_dims['object_hue']['nbr_fillers'] = int(strategy[7].split('FP')[0])

                self.latent_dims['object_hue']['position'] = 2
                # 8: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 10  
                if 'RemainderToUse' in strategy[8]:
                    strategy[8] = strategy[8].split('RemainderToUse')
                    self.latent_dims['object_hue']['remainder_use'] = int(strategy[8][1])
                    strategy[8] = strategy[8][0]
                else:
                    self.latent_dims['object_hue']['remainder_use'] = 0
                self.latent_dims['object_hue']['divider'] = int(strategy[8])
                # 9: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 and 10 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[9]:
                    self.latent_dims['object_hue']['untested'] = True
                    self.latent_dims['object_hue']['test_set_divider'] = (self.latent_dims['object_hue']['size']//self.latent_dims['object_hue']['divider'])+10
                elif 'E' in strategy[9]:  
                    self.latent_dims['object_hue']['test_set_size_sample_from_end'] = int(strategy[9][1:])
                elif 'S' in strategy[9]:  
                    self.latent_dims['object_hue']['test_set_size_sample_from_start'] = int(strategy[9][1:])
                else:
                    self.latent_dims['object_hue']['test_set_divider'] = int(strategy[9])
                
                # 10: Scale
                self.latent_dims['Scale'] = {'size': 8}
                
                self.latent_dims['Scale']['nbr_fillers'] = 0
                self.latent_dims['Scale']['primitive'] = ('FP' in strategy[10])
                if self.latent_dims['Scale']['primitive']:
                    self.latent_dims['Scale']['nbr_fillers'] = int(strategy[10].split('FP')[0])

                self.latent_dims['Scale']['position'] = 3
                # 11: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 6  
                if 'RemainderToUse' in strategy[11]:
                    strategy[11] = strategy[11].split('RemainderToUse')
                    self.latent_dims['Scale']['remainder_use'] = int(strategy[11][1])
                    strategy[11] = strategy[11][0]    
                else:
                    self.latent_dims['Scale']['remainder_use'] = 0
                self.latent_dims['Scale']['divider'] = int(strategy[11])
                # 12: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[12]:
                    self.latent_dims['Scale']['untested'] = True
                    self.latent_dims['Scale']['test_set_divider'] = (self.latent_dims['Scale']['size']//self.latent_dims['Scale']['divider'])+10
                elif 'E' in strategy[12]:  
                    self.latent_dims['Scale']['test_set_size_sample_from_end'] = int(strategy[12][1:])
                elif 'S' in strategy[12]:  
                    self.latent_dims['Scale']['test_set_size_sample_from_start'] = int(strategy[12][1:])
                else:
                    self.latent_dims['Scale']['test_set_divider'] = int(strategy[12])

                # 13: Shape
                self.latent_dims['Shape'] = {'size': 4}
                
                self.latent_dims['Shape']['nbr_fillers'] = 0
                self.latent_dims['Shape']['primitive'] = ('FP' in strategy[13])
                if self.latent_dims['Shape']['primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[13].split('FP')[0])

                self.latent_dims['Shape']['position'] = 4
                # 14: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 3  
                if 'RemainderToUse' in strategy[14]:
                    strategy[14] = strategy[14].split('RemainderToUse')
                    self.latent_dims['Shape']['remainder_use'] = int(strategy[14][1])
                    strategy[14] = strategy[14][0]    
                else:
                    self.latent_dims['Shape']['remainder_use'] = 0
                self.latent_dims['Shape']['divider'] = int(strategy[14])
                # 15: test_set_divider (default:3) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=3 => effective indices 3 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[15]:
                    self.latent_dims['Shape']['untested'] = True
                    self.latent_dims['Shape']['test_set_divider'] = (self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'])+10
                elif 'E' in strategy[15]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_end'] = int(strategy[15][1:])
                elif 'S' in strategy[15]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_start'] = int(strategy[15][1:])
                else:
                    self.latent_dims['Shape']['test_set_divider'] = int(strategy[15])
                
                # 16: Orientation
                self.latent_dims['Orientation'] = {'size': 15}
                
                self.latent_dims['Orientation']['nbr_fillers'] = 0
                self.latent_dims['Orientation']['primitive'] = ('FP' in strategy[16])
                if self.latent_dims['Orientation']['primitive']:
                    self.latent_dims['Orientation']['nbr_fillers'] = int(strategy[16].split('FP')[0])

                self.latent_dims['Orientation']['position'] = 5
                # 17: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 3  
                if 'RemainderToUse' in strategy[17]:
                    strategy[17] = strategy[17].split('RemainderToUse')
                    self.latent_dims['Orientation']['remainder_use'] = int(strategy[17][1])
                    strategy[17] = strategy[17][0]    
                else:
                    self.latent_dims['Orientation']['remainder_use'] = 0
                self.latent_dims['Orientation']['divider'] = int(strategy[17])
                # 18: test_set_divider (default:3) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=3 => effective indices 3 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[18]:
                    self.latent_dims['Orientation']['untested'] = True
                    self.latent_dims['Orientation']['test_set_divider'] = (self.latent_dims['Orientation']['size']//self.latent_dims['Orientation']['divider'])+10
                elif 'E' in strategy[18]:  
                    self.latent_dims['Orientation']['test_set_size_sample_from_end'] = int(strategy[18][1:])
                elif 'S' in strategy[18]:  
                    self.latent_dims['Orientation']['test_set_size_sample_from_start'] = int(strategy[18][1:])
                else:
                    self.latent_dims['Orientation']['test_set_divider'] = int(strategy[18])

                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)

        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        self.traintest_indices = []
        if self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(len(self.imgs)):
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
            print(f"Dataset Size: {len(self.indices)} out of 737280: {100*len(self.indices)/737280}%.")
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
            print(f"Dataset Size: {len(self.indices)} out of 480000 : {100*len(self.indices)/480000}%.")
            

        self.latents_one_hot_encodings = np.eye(15)[self.latents_classes.reshape(-1)]
        #self.latents_one_hot_encodings = self.latents_one_hot_encodings.reshape((-1, 6, 40))
        self.latents_one_hot_encodings = self.latents_one_hot_encodings.reshape((-1, 6*15))
        
        """
        self.imgs = self.imgs[self.indices]
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        
        self.test_latents_mask = self.test_latents_mask[self.indices]
        self.targets = self.targets[self.indices]
        """

        self.imgs = self.imgs[self.traintest_indices]
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
        images = [Image.fromarray(im, mode='RGB') for im in self.imgs[self.factors_indices]]
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
        image = Image.fromarray(self.imgs[trueidx], mode='RGB')

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
