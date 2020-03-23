from typing import Dict, List, Tuple
from .dataset import Dataset
import torch
import random


class DualLabeledDataset(Dataset):
    def __init__(self, kwargs):
        super(DualLabeledDataset, self).__init__(kwargs)
        self.kwargs = kwargs
        self.datasets = {'train':kwargs['train_dataset'],
                            'test':kwargs['test_dataset']
                            }
        self.mode = kwargs['mode']

        self.train_classes = {}
        for idx in range(len(self.datasets['train'])):
            if hasattr(self.datasets['train'], 'getclass'):
                cl = self.datasets['train'].getclass(idx)
            else :
                _, cl = self.datasets['train'][idx]
            if cl not in self.train_classes: self.train_classes[cl] = []
            self.train_classes[cl].append(idx)

        test_idx_offset = len(self.datasets['train'])
        self.test_classes = {}
        for idx in range(len(self.datasets['test'])):
            if hasattr(self.datasets['test'], 'getclass'):
                cl = self.datasets['test'].getclass(idx)
            else :
                _, cl = self.datasets['test'][idx]
            if cl not in self.test_classes: self.test_classes[cl] = []
            self.test_classes[cl].append(test_idx_offset+idx)

        # Adding the train classes to the test classes so that we can sample
        # distractors from the train set:
        for cl in self.train_classes:
            if cl not in self.test_classes:
                self.test_classes[cl] = []
            for idx in self.train_classes[cl]:
                self.test_classes[cl].append(idx)

        self.nbr_classes = len(self.test_classes.keys())
    
    def set_mode(self, newmode='train'):
        self.mode = newmode

    def __len__(self) -> int:
        if self.mode == 'test':
            return len(self.datasets['test'])
        else:
            return len(self.datasets['train'])
    
    def getNbrClasses(self) -> int:
        if self.mode =='test':
            return self.nbr_classes
        else:
            return len(self.train_classes)

    def sample(self, idx: int = None, from_class: List[int] = None, excepts: List[int] = None, target_only: bool = False) -> Tuple[torch.Tensor, List[int]]:
        '''
        This type of dataset can only be used in non-descriptive mode!
        Thus, `idx` is never `None`

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
            - exp_latents: 
            - exp_latents_values: 
        '''
        assert(idx is not None)

        classes = self.train_classes 
        if self.mode == 'test':
            classes = self.test_classes
            idx += len(self.datasets['train'])

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
        
        experiences = []
        exp_labels = []
        exp_latents = []
        exp_latents_values = []
        for idx in indices:
            dataset = self.datasets['train']
            if idx>=len(self.datasets['train']):
                dataset = self.datasets['test']
                idx -= len(self.datasets['train'])

            sample_output = dataset[idx]
            if len(sample_output) == 2:
                exp, tc = sample_output
                if isinstance(tc, int): 
                    #latent = torch.Tensor([tc])
                    latent = torch.zeros((self.nbr_classes))
                    latent[tc] = 1.0
                    latent_values = latent
            elif len(sample_output) == 3:
                exp, tc, latent = sample_output
                if isinstance(latent, int): latent = torch.Tensor([latent])
                if isinstance(tc, int): 
                    #latent = torch.Tensor([tc])
                    latent_values = torch.zeros((self.nbr_classes))
                    latent_values[tc] = 1.0
            elif len(sample_output) == 4:
                exp, tc, latent, latent_values = sample_output
                if isinstance(latent, int): latent = torch.Tensor([latent])
                if isinstance(latent_values, int): latent_values = torch.Tensor([latent_values])
            else:
                raise NotImplemented
            experiences.append(exp.unsqueeze(0))
            exp_labels.append(tc)
            exp_latents.append(latent.unsqueeze(0))
            exp_latents_values.append(latent_values.unsqueeze(0))
            if target_only: break

        experiences = torch.cat(experiences,dim=0)
        experiences = experiences.unsqueeze(1)
        exp_latents = torch.cat(exp_latents, dim=0)
        exp_latents_values = torch.cat(exp_latents_values, dim=0)
        #exp_latents = exp_latents.unsqueeze(1)
        # account for the temporal dimension...
        
        return experiences, indices, exp_labels, exp_latents, exp_latents_values
