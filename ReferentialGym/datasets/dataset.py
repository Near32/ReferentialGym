from typing import Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset as torchDataset

class Dataset(torchDataset):
    def __init__(self, kwargs):
        '''
        :attribute classes: List (or Dictionary) of Lists of (absolute) indices of stimulus.
        '''
        super(Dataset,self).__init__()

        self.kwargs = kwargs
        if "root_folder" in kwargs:
            self.root_folder = kwargs['root_folder']
        self.classes = None 

    def __len__(self) -> int:
        raise NotImplementedError

    def getNbrClasses(self) -> int:
        raise NotImplementedError

    def sample(self, idx: int = None, from_class: List[int] = None, excepts: List[int] = None) -> Tuple[torch.Tensor, List[int]]:
        '''
        Sample a stimulus from the dataset.
        If :param from_class: is not None, the sampled stimulus will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :returns:
            - stimuli: Tensor of the sampled stimuli.
            - indices: List[int] of the indices of the sampled stimuli.
        '''
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        '''
        Samples target and distractors (uniformly).
        
        :params idx: int, index of the stimulus to use as a target.
        :returns: 
            - stimuli: Tensor of shape (1, nbr_distractors+1, nbr_stimulus, stimulus_shape)
        '''
        return self.sample(idx=idx)[0]
