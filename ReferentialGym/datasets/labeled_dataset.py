from typing import Dict, List, Tuple
from .dataset import Dataset
import torch
import random


class LabeledDataset(Dataset):
    def __init__(self, kwargs):
        super(LabeledDataset, self).__init__(kwargs)
        self.kwargs = kwargs
        self.dataset = kwargs['dataset']
        
        self.nbr_distractors = self.kwargs['nbr_distractors']
        self.nbr_stimulus = self.kwargs['nbr_stimulus']
        
        self.classes = {}
        for idx in range(len(self.dataset)):
            img, cl = self.dataset[idx]
            if cl not in self.classes: self.classes[cl] = []
            self.classes[cl].append(idx)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def getNbrClasses(self) -> int:
        return len(self.classes)

    def sample(self, idx: int = None, from_class: List[int] = None, excepts: List[int] = None, target_only: bool = False) -> Tuple[torch.Tensor, List[int]]:
        '''
        Sample an experience from the dataset. Along with relevant distractor experiences.
        If :param from_class: is not None, the sampled experiences will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :param target_only: bool (default: `False`) defining whether to sample only the target or distractors too.

        :returns:
            - experiences: Tensor of the sampled experiences.
            - indices: List[int] of the indices of the sampled experiences.
            - exp_labels: List[int] consisting of the indices of the label to which the experiences belong.
        '''
        if from_class is None:
            from_class = range(10)
            
        set_indices = set()
        for class_idx in from_class:
            set_indices = set_indices.union(set(self.classes[class_idx]))
        
        if excepts is not None:
            set_indices = set_indices.difference(excepts)
            
        indices = []
        nbr_samples = self.nbr_distractors
        if idx is None: 
            nbr_samples += 1
        else: 
            try:
                set_indices.remove(idx)
            except Exception as e:
                print("Exception caught during removal of the target index:")
                print(e)
            indices.append(idx)

        for choice_idx in range(nbr_samples):
            chosen = random.choice( list(set_indices))
            set_indices.remove(chosen)
            indices.append( chosen)
        
        experiences = []
        exp_labels = []
        for idx in indices:
            exp, tc = self.dataset[idx]
            experiences.append(exp.unsqueeze(0))
            exp_labels.append(tc)
            if target_only and idx==0: break

        experiences = torch.cat(experiences,dim=0)
        experiences = experiences.unsqueeze(1)
        # account for the temporal dimension...
        
        return experiences, indices, exp_labels