from typing import Dict, List, Tuple
import ReferentialGym
from ReferentialGym.datasets.dataset import Dataset
import torch
import numpy as np
import random


class DualLabeledDataset(Dataset):
    def __init__(self, kwargs):
        super(DualLabeledDataset, self).__init__(kwargs)
        self.kwargs = kwargs
        self.datasets = {'train':kwargs['train_dataset'],
                            'test':kwargs['test_dataset']
                            }
        self.mode = kwargs['mode']
        self.mode2offsetidx2class = {'train':{}, 'test':{}}
        self.with_replacement= kwargs["with_replacement"]

        self.train_classes = {}
        for idx in range(len(self.datasets['train'])):
            if hasattr(self.datasets['train'], 'getclass'):
                cl = self.datasets['train'].getclass(idx)
            else :
                _, cl = self.datasets['train'][idx]
            if cl not in self.train_classes: self.train_classes[cl] = []
            self.train_classes[cl].append(idx)
            self.mode2offsetidx2class['train'][idx] = cl

        test_idx_offset = len(self.datasets['train'])
        self.test_classes = {}
        for idx in range(len(self.datasets['test'])):
            if hasattr(self.datasets['test'], 'getclass'):
                cl = self.datasets['test'].getclass(idx)
            else :
                _, cl = self.datasets['test'][idx]
            if cl not in self.test_classes: self.test_classes[cl] = []
            self.test_classes[cl].append(test_idx_offset+idx)
            self.mode2offsetidx2class['test'][test_idx_offset+idx] = cl

        # Adding the train classes to the test classes so that we can sample
        # distractors from the train set:
        for cl in self.train_classes:
            if cl not in self.test_classes:
                self.test_classes[cl] = []
            for idx in self.train_classes[cl]:
                self.test_classes[cl].append(idx)
                self.mode2offsetidx2class['test'][idx] = cl

        self.nbr_classes = len(self.test_classes.keys())
        
        # Per-Target Distributional Distractor Sampling Scheme : 
        # initialized uniformely.
        train_test_size = sum([len(d) for d in self.datasets.values()])
        self.distractor_sampling_likelihoods = np.ones((train_test_size, train_test_size))
    
    def _get_class_from_idx(self, idx):
        '''
        This function expects an index that may be offset
        by the length of the training dataset, which 
        means that it is a at-test-time sampling.
        This function then checks the class with a 
        regularised index.
        '''
        dataset = self.datasets['train']
        sampling_idx = idx 
        if sampling_idx>=len(dataset):
            dataset = self.datasets['test']
            sampling_idx -= len(self.datasets['train'])

        if hasattr(dataset, 'getclass'):
            cl = dataset.getclass(sampling_idx)
        else :
            _, cl = dataset[sampling_idx]
        
        return cl 

    def set_mode(self, newmode='train'):
        self.mode = newmode

    def size(self) -> int:
        if 'test' in self.mode:
            return self.datasets['test'].size()
        else:
            return self.datasets['train'].size()
    
    def __len__(self) -> int:
        if 'test' in self.mode:
            return len(self.datasets['test'])
        else:
            return len(self.datasets['train'])
    
    def getNbrClasses(self) -> int:
        if 'test' in self.mode:
            return self.nbr_classes
        else:
            return len(self.train_classes)

    def sample(self, 
               idx: int = None, 
               from_class: List[int] = None, 
               excepts: List[int] = None, 
               excepts_class: List[int]=None, 
               target_only: bool = False) -> Dict[str,object]:
        '''
        Sample an experience from the dataset. Along with relevant distractor experiences.
        If :param from_class: is not None, the sampled experiences will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes
                            to identifies classes to sample from.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :param excepts_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes
                            to identifies classes to not sample from.
        :param target_only: bool (default: `False`) defining whether to sample only the target or distractors too.

        :returns:
            - sample_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"indices"`: List[int] of the indices of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
                - `"exp_latents"`: Tensor representatin the latent of the experience in one-hot-encoded vector form.
                - `"exp_latents_values"`: Tensor representatin the latent of the experience in value form.
                - some other keys provided by the dataset used...
        '''
        classes = self.train_classes 
        idx_offset = 0 
        if 'test' in self.mode:
            classes = self.test_classes
            idx_offset = len(self.datasets['train'])

        DSS_version = ReferentialGym.datasets.dataset.DSS_version
        if DSS_version == 1:
            test = True
            not_enough_elements = False
            while test:
                if from_class is None or not_enough_elements:
                    from_class = set(classes.keys())
             
                # If object_centric, then make sure the distractors
                # are not sampled from the target's class:
                if idx is not None and self.kwargs['object_centric']:
                    # Expect an offset index if testing:
                    class_of_idx = self._get_class_from_idx(idx+idx_offset)
                    if class_of_idx in from_class:
                        from_class.remove(class_of_idx)
    
                list_indices = []
                for class_idx in from_class:
                    list_indices += classes[class_idx]
                set_indices = set(list_indices) 
                # these indices contain the offset.

                if excepts_class is not None:
                    excepts_list_indices = []
                    for class_idx in excepts_class:
                        excepts_list_indices += classes[class_idx]
                    set_indices = set_indices.difference(set(excepts_list_indices))
             
                if excepts is not None:
                    # check that the current class contains more than just one element:
                    if len(set_indices) != 1:
                        set_indices = set_indices.difference(excepts)
                 
                indices = []
                nbr_samples = 1
                if not target_only:
                    nbr_samples += self.nbr_distractors[self.mode]
    
                if idx is not None:
                    # i.e. if we are not trying to resample the target stimulus...
                    if idx+idx_offset in set_indices:
                        set_indices.remove(idx+idx_offset)
                    indices.append(idx+idx_offset)
             
                if len(set_indices) < nbr_samples:
                    #print("WARNING: Dataset's class has not enough element to choose from...")
                    #print("WARNING: Using all the classes to sample...")
                    not_enough_elements = True
                else:
                    test = False 
        elif DSS_version==2:
            test = True
            not_enough_elements = False
            while test:
                if from_class is None or not_enough_elements:
                    from_class = set(classes.keys())
             
                # If object_centric, then make sure the distractors
                # are not sampled from the target's class:
                if idx is not None and self.kwargs['object_centric']:
                    # Expect an offset index if testing:
                    class_of_idx = self._get_class_from_idx(idx+idx_offset)
                    if class_of_idx in from_class:
                        from_class.remove(class_of_idx)
    
                list_indices = []
                for class_idx in from_class:
                    list_indices += classes[class_idx]
                set_indices = set(list_indices) 
                # these indices contain the offset.
    
                if excepts_class is not None:
                    excepts_list_indices = []
                    for class_idx in excepts_class:
                        excepts_list_indices += classes[class_idx]
                    set_indices = set_indices.difference(set(excepts_list_indices))
             
                if excepts is not None:
                    # check that the current class contains more than just one element:
                    if len(set_indices) != 1:
                        set_indices = set_indices.difference(excepts)
                 
                indices = []
                nbr_samples = 1
                if not target_only:
                    nbr_samples += self.nbr_distractors[self.mode]
    
                if idx is not None:
                    # i.e. if we are not trying to resample the target stimulus...
                    #if idx in set_indices:
                    if idx+idx_offset in set_indices:
                        set_indices.remove(idx+idx_offset)
                    indices.append(idx+idx_offset)
             
                if not self.with_replacement \
                and len(set_indices) < nbr_samples:
                    not_enough_elements = True
                elif self.with_replacement \
                and len(set_indices) < 2:
                    #print("WARNING: Dataset's class has not enough element to choose from...")
                    #print("WARNING: Using all the classes to sample...")
                    not_enough_elements = True
                else:
                    test = False 
        else:
            raise NotImplementedError 
        
        if idx is not None:
            target_idx = idx+idx_offset
        else:
            #TODO: need to figure out when does this occur:
            # this occurs when trying to sample listener distractor for DC_version==2 
            # from Dataset __getitem__ method.
            # What can we use as the target_idx to sample from then?
            # Anything from the current class sounds OK?
            assert from_class is not None
            possible_target_list = []
            for class_idx in from_class:
                possible_target_list += classes[class_idx]
            target_idx = np.random.choice(possible_target_list)

        while len(indices) < nbr_samples:
            list_indices = list(set_indices)
            list_likelihoods = self.distractor_sampling_likelihoods[target_idx, list_indices]
            norm = np.sum(list_likelihoods)
            if norm!=0 :
                list_norm_likelihoods = list_likelihoods/norm
            else:
                list_norm_likelihoods = np.ones_like(list_likelihoods)/len(list_likelihoods)
            chosen = np.random.choice(
                a=list_indices,
                p=list_norm_likelihoods,
            )
            if not self.with_replacement:
                set_indices.remove(chosen)
            indices.append(chosen)
        
        sample_d = {
            "experiences":[],
            "exp_labels":[],
            "exp_latents":[],
            "exp_latents_values":[]
        }

        for idx_with_offset in indices:
            idx = idx_with_offset - idx_offset
            need_reg = {k:True for k in sample_d}

            dataset = self.datasets['train']
            if idx_with_offset>=len(self.datasets['train']):
                dataset = self.datasets['test']
                sampled_d = dataset[idx]
            else:
                sampled_d = dataset[idx_with_offset]
            
            for key, value in sampled_d.items():
                if key not in sample_d:
                    sample_d[key] = []
                else:
                    need_reg[key] = False
                
                if key=="exp_labels":
                    current_mode = 'train'
                    if 'test' in self.mode: current_mode = 'test'
                    value = self.mode2offsetidx2class[current_mode][idx_with_offset]
                sample_d[key].append(value)

            # We assume that it is a supervision learning dataset,
            # therefore it ought to have labels...
            assert(need_reg["exp_labels"] == False)

            if need_reg["exp_latents"]:
                assert(isinstance(sampled_d["exp_labels"], int))
                latent = torch.zeros((self.nbr_classes))
                latent[sampled_d["exp_labels"]] = 1.0
                
                sample_d["exp_latents"].append(latent)
                need_reg["exp_latents"] = False

            if need_reg["exp_latents_values"]:
                sample_d["exp_latents_values"] = sample_d["exp_latents"]
                need_reg["exp_latents_values"] = False

            if target_only: break

        for key, lvalue in sample_d.items():
            if not(isinstance(lvalue[0], torch.Tensor)):    continue
            sample_d[key] = torch.stack(lvalue)

        # Add the stimulus size / temporal dimension:
        for k,v in sample_d.items():
            if not(isinstance(v, torch.Tensor)):    continue
            sample_d[k] = v.unsqueeze(1)
        
        # Adding the sampled indices, with their offset:
        sample_d["indices"] = indices 

        return sample_d
