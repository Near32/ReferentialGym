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

    def sample(self, idx: int = None, from_class: List[int] = None, excepts: List[int] = None) -> Tuple[torch.Tensor, List[int]]:
        '''
        Sample a stimulus from the dataset.
        If :param from_class: is not None, the sampled stimulus will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions indices.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :returns:
            -stimuli: Tensor of the sampled stimuli.
            -indices: List[int] of the indices of the sampled stimuli.
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
            try:
                set_indices = set_indices.remove(idx)
            except Exception as e:
                print("Exception caught during removal of the target index:")
                print(e)
        else: indices.append(idx)
        for _ in range(nbr_samples):
            chosen = random.choice( list(set_indices))
            set_indices.remove(chosen)
            indices.append( chosen)
        
        stimuli = []
        for idx in indices:
            st, tc = self.dataset[idx]
            stimuli.append(st.unsqueeze(0))
            
        stimuli = torch.cat(stimuli,dim=0)
        stimuli = stimuli.unsqueeze(1)
        # account for the temporal dimension...
        
        return stimuli, indices
    
    def __getitem__(self, idx):
        '''
        Samples target and distractors (uniformly).
        
        :params idx: int, index of the stimulus to use as a target.
        :returns: 
            - stimuli: Tensor of shape (nbr_distractors+1, nbr_stimulus, stimulus_shape)
        '''
        return self.sample(idx)[0]