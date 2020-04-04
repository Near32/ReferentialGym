from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset 
import os
import numpy as np
import random
from PIL import Image 


class dSpritesDataset(Dataset) :
    def __init__(self, root='./', train=True, transform=None, split_strategy=None) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.train = train
        self.root = os.path.join(root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        self.transform = transform
        self.split_strategy = split_strategy


        # Load dataset
        dataset_zip = np.load(self.root, encoding='latin1', allow_pickle=True)
        #data = np.load(root, encoding='latin1')
        #data = torch.from_numpy(data['imgs']).unsqueeze(1).float())
        print('Keys in the dataset:')
        for k in dataset_zip.keys(): print(k)
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.targets = np.zeros(len(self.latents_classes)) #[random.randint(0, 10) for _ in self.imgs]
        for idx, latent_cls in enumerate(self.latents_classes):
            posX = latent_cls[-2]
            posY = latent_cls[-1]
            target = posX*32+posY
            self.targets[idx] = target  
        self.metadata = dataset_zip['metadata'][()]
        
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
                # 1: Y
                self.latent_dims['Y'] = {'size': 32}
                
                self.latent_dims['Y']['nbr_fillers'] = 0
                self.latent_dims['Y']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['Y']['primitive']:
                    self.latent_dims['Y']['nbr_fillers'] = int(strategy[1].split('FP')[0])

                self.latent_dims['Y']['position'] = 5
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                self.latent_dims['Y']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['Y']['untested'] = True
                    self.latent_dims['Y']['test_set_divider'] = (self.latent_dims['Y']['size']//self.latent_dims['Y']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['Y']['test_set_divider'] = int(strategy[3])

                # 4: X
                self.latent_dims['X'] = {'size': 32}
                
                self.latent_dims['X']['nbr_fillers'] = 0
                self.latent_dims['X']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['X']['primitive']:
                    self.latent_dims['X']['nbr_fillers'] = int(strategy[4].split('FP')[0])

                self.latent_dims['X']['position'] = 4
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                self.latent_dims['X']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['X']['untested'] = True
                    self.latent_dims['X']['test_set_divider'] = (self.latent_dims['X']['size']//self.latent_dims['X']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['X']['test_set_divider'] = int(strategy[6])
                # 7: Orientation
                self.latent_dims['Orientation'] = {'size': 40}
                
                self.latent_dims['Orientation']['nbr_fillers'] = 0
                self.latent_dims['Orientation']['primitive'] = ('FP' in strategy[7])
                if self.latent_dims['Orientation']['primitive']:
                    self.latent_dims['Orientation']['nbr_fillers'] = int(strategy[7].split('FP')[0])

                self.latent_dims['Orientation']['position'] = 3
                # 8: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 10  
                self.latent_dims['Orientation']['divider'] = int(strategy[8])
                # 9: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 and 10 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[9]:
                    self.latent_dims['Orientation']['untested'] = True
                    self.latent_dims['Orientation']['test_set_divider'] = (self.latent_dims['Orientation']['size']//self.latent_dims['Orientation']['divider'])+10
                else:  
                    self.latent_dims['Orientation']['test_set_divider'] = int(strategy[9])
                
                # 10: Scale
                self.latent_dims['Scale'] = {'size': 6}
                
                self.latent_dims['Scale']['nbr_fillers'] = 0
                self.latent_dims['Scale']['primitive'] = ('FP' in strategy[10])
                if self.latent_dims['Scale']['primitive']:
                    self.latent_dims['Scale']['nbr_fillers'] = int(strategy[10].split('FP')[0])

                self.latent_dims['Scale']['position'] = 2
                # 11: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 6  
                self.latent_dims['Scale']['divider'] = int(strategy[11])
                # 12: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[12]:
                    self.latent_dims['Scale']['untested'] = True
                    self.latent_dims['Scale']['test_set_divider'] = (self.latent_dims['Scale']['size']//self.latent_dims['Scale']['divider'])+10
                else:  
                    self.latent_dims['Scale']['test_set_divider'] = int(strategy[12])
                
                # 13: Shape
                self.latent_dims['Shape'] = {'size': 3}
                
                self.latent_dims['Shape']['nbr_fillers'] = 0
                self.latent_dims['Shape']['primitive'] = ('FP' in strategy[13])
                if self.latent_dims['Shape']['primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[13].split('FP')[0])

                self.latent_dims['Shape']['position'] = 1
                # 14: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 3  
                self.latent_dims['Shape']['divider'] = int(strategy[14])
                # 15: test_set_divider (default:3) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=3 => effective indices 3 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[15]:
                    self.latent_dims['Shape']['untested'] = True
                    self.latent_dims['Shape']['test_set_divider'] = (self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'])+10
                else:  
                    self.latent_dims['Shape']['test_set_divider'] = int(strategy[15])
                
                # COLOR: TODO...

                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]])
                assert(nbr_primitives_and_tested==self.counter_test_threshold)

        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        if self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(len(self.imgs)):
                if idx % self.divider == self.offset:
                    self.indices.append(idx)

            self.train_ratio = 0.8
            end = int(len(self.indices)*self.train_ratio)
            if self.train:
                self.indices = self.indices[:end]
            else:
                self.indices = self.indices[end:]

            print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
            print(f"Dataset Size: {len(self.indices)} out of 737280: {100*len(self.indices)/737280}%.")
        elif 'combinatorial' in self.split_strategy:
            indices_latents = list(zip(range(self.latents_classes.shape[0]), self.latents_classes))
            for idx, latent_class in indices_latents:
                effective_test_threshold = self.counter_test_threshold
                counter_test = {}
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[dim_dict['position']]
                    quotient = (dim_class+1)//dim_dict['divider']
                    remainder = (dim_class+1)%dim_dict['divider']
                    if remainder!=0:
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

                if skip_it: continue


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
            print(self.latent_dims)
            print(f"Dataset Size: {len(self.indices)} out of 737280 : {100*len(self.indices)/737280}%.")
            


        self.imgs = self.imgs[self.indices]
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        self.targets = self.targets[self.indices]
        del self.metadata

        print('Dataset loaded : OK.')
        
    def __len__(self) -> int:
        return len(self.indices)

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        #idx = self.indices[idx]
        #target = self.latents_classes[idx]
        target = self.targets[idx]
        return target

    def getlatentvalue(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        #idx = self.indices[idx]
        latent_value = self.latents_values[idx]
        return latent_value

    def getlatentclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        #idx = self.indices[idx]
        latent_class = self.latents_classes[idx]
        return latent_class

    def __getitem__(self, idx:int) -> Dict[str,torch.Tensor]:
        """
        :param idx: Integer index.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representatin the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representatin the latent of the experience in value form.
        """
        if idx >= len(self):
            idx = idx%len(self)
        #orig_idx = idx
        #idx = self.indices[idx]

        #img, target = self.dataset[idx]
        image = Image.fromarray((self.imgs[idx]*255).astype('uint8'))
        
        #target = self.getclass(orig_idx)
        #latent_value = torch.from_numpy(self.getlatentvalue(orig_idx))
        
        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_class = torch.from_numpy(self.getlatentclass(idx))
        
        if self.transform is not None:
            image = self.transform(image)
        
        sampled_d = {
            "experiences":image, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value
        }

        return sampled_d