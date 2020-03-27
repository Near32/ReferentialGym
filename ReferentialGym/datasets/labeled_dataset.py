from typing import Dict, List, Tuple
from .dataset import Dataset
import torch
import random


class LabeledDataset(Dataset):
    def __init__(self, kwargs):
        super(LabeledDataset, self).__init__(kwargs)
        self.kwargs = kwargs
        self.dataset = kwargs['dataset']
        self.mode = kwargs['mode']

        self.classes = {}
        for idx in range(len(self.dataset)):
            if hasattr(self.dataset, 'getclass'):
                cl = self.dataset.getclass(idx)
            else :
                _, cl = self.dataset[idx]
            if cl not in self.classes: self.classes[cl] = []
            self.classes[cl].append(idx)

        self.nbr_classes = len(self.classes.keys())
    
    def set_mode(self, newmode='train'):
        self.mode = newmode
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def getNbrClasses(self) -> int:
        return len(self.classes)

    def sample(self, 
               idx: int = None, 
               from_class: List[int] = None, 
               excepts: List[int] = None, 
               target_only: bool = False) -> Dict[str,object]:
        '''
        Sample an experience from the dataset. Along with relevant distractor experiences.
        If :param from_class: is not None, the sampled experiences will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
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
        test = True
        not_enough_elements = False
        while test:
            if from_class is None or not_enough_elements:
                from_class = list(classes.keys())
                
            set_indices = set()
            for class_idx in from_class:
                set_indices = set_indices.union(set(classes[class_idx]))
            
            if excepts is not None:
                set_indices = set_indices.difference(excepts)
                
            indices = []
            nbr_samples = self.nbr_distractors[self.mode]
            try:
                set_indices.remove(idx)
            except Exception as e:
                print("Exception caught during removal of the target index:")
                print(e)
                import ipdb; ipdb.set_trace()
            indices.append(idx)

            if len(set_indices) < nbr_samples:
                print("WARNING: Dataset's class has not enough element to choose from...")
                print("WARNING: Using all the classes to sample...")
                not_enough_elements = True
            else:
                test = False 

        for choice_idx in range(nbr_samples):
            chosen = random.choice( list(set_indices))
            set_indices.remove(chosen)
            indices.append( chosen)

        sample_d = {
            "experiences":[],
            "exp_labels":[],
            "exp_latents":[],
            "exp_latents_values":[]
        }

        for idx in indices:
            need_reg = {k:True for k in sample_d}

            sampled_d = self.dataset[idx]
            for key, value in sampled_d.items():
                if key not in sample_d:
                    sample_d[key] = []
                else:
                    need_reg[key] = False
                
                if isinstance(value, torch.Tensor): 
                    value = value.unsqueeze(0)
                
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
            sample_d[key] = torch.cat(lvalue, dim=0)

        # Add the stimulus size / temporal dimension:
        sample_d["experiences"] = sample_d["experiences"].unsqueeze(1)
        
        # Adding the sampled indices:
        sample_d["indices"] = indices 

        return sample_d
