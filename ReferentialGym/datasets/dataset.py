from typing import Dict, List, Integer
import numpy as np 

class Dataset():
    def __init__(self, root_folder):
        '''
        :attribute classes: List (or Dictionary) of Lists of (absolute) indices of stimulus.
        '''
        self.root_folder = root_folder
        self.classes = None 

    def __len__(self) -> Integer:
        raise NotImplementedError

    def getNbrClasses(self) -> Integer:
        raise NotImplementedError

    def sample(self, idx, from_class: List[Integer] = None, excepts: List[Integer] = None) -> np.array:
        '''
        Sample a stimulus from the dataset.
        If :param from_class: is not None, the sampled stimulus will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :return sampled_stimulus: Numpy Array of the sampled stimulus.
        '''
        raise NotImplementedError