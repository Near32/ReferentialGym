from typing import Dict

import torch
from torch.utils.data import Dataset 


class DictDatasetWrapper(Dataset) :
    def __init__(self, dataset) :
        '''
        :param dataset: Dataset to wrap... 
        '''
        self.dataset = dataset
        
    def __len__(self) :
        return len(self.dataset)

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        if hastattr(self.dataset, "getclass"):
            return self.dataset.getclass(idx)
        else:    
            _, label = self.dataset[idx]
            return label

    def __getitem__(self, idx:int) -> Dict[str,torch.Tensor]:
        """
        :param idx: Integer index.

        :returns:
            sampled_d: Dict of:
                - `"experiences"`: Tensor of the sampled experiences.
                - `"exp_labels"`: List[int] consisting of the indices of the label to which the experiences belong.
        """
        if idx >= len(self):
            idx = idx%len(self)
        
        exp, label = self.dataset[idx]
        
        sampled_d = {
            "experiences":exp, 
            "exp_labels":label, 
        }

        return sampled_d