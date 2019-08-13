from typing import Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset as torchDataset

def shuffle(experiences):
    st_size = experiences.size()
    batch_size = st_size[0]
    nbr_distractors_po = st_size[1]
    perms = []
    shuffled_experiences = []
    for b in range(batch_size):
        perm = torch.randperm(nbr_distractors_po)
        if experiences.is_cuda: perm = perm.cuda()
        perms.append(perm.unsqueeze(0))
        shuffled_experiences.append( experiences[b,perm,...].unsqueeze(0))
    perms = torch.cat(perms, dim=0)
    shuffled_experiences = torch.cat(shuffled_experiences, dim=0)
    decision_target = (perms==0).max(dim=1)[1].long()
    return shuffled_experiences, decision_target


class Dataset(torchDataset):
    def __init__(self, kwargs):
        '''
        :attribute classes: List (or Dictionary) of Lists of (absolute) indices of experiences.
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
        Sample an experience from the dataset. Along with relevant distractor experiences.
        If :param from_class: is not None, the sampled experiences will belong to the specified class(es).
        If :param excepts: is not None, this function will make sure to not sample from the specified list of exceptions.
        :param from_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :returns:
            - experiences: Tensor of the sampled experiences.
            - indices: List[int] of the indices of the sampled experiences.
        '''
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        '''
        Samples target experience and distractor experiences according to the distractor sampling scheme.
        
        :params idx: int, index of the experiences to use as a target.
        :returns: 
            - 'speaker_experiences': Tensor of shape `(1, A, nbr_stimulus, stimulus_shape)` where `A=nbr_distractors+1` 
                                  if `self.kwargs['observability']=='full'` or `A=1` otherwise (`'partial'`).
            - 'listener_experiences': Tensor of shape `(1, nbr_distractors+1, nbr_stimulus, stimulus_shape)`.
            - 'target_decision_idx': Tensor of type Long and shape `(1,)` containing the index of the target experience
                                     among the 'listener_experiences'.
        '''
        if self.kwargs['descriptive']:
            experiences = self.sample(idx=idx)[0].unsqueeze(0)
            # In descriptive mode, the speaker observability is alwasy partial:
            speaker_experiences = experiences[:,0].unsqueeze(1)
            
            if torch.rand(size=(1,)).item() > self.kwargs['descriptive_target_ratio']:
                # Target experience remain in the experiences yielded to the listener:
                listener_experiences = experiences 
                listener_experiences, target_decision_idx = shuffle(listener_experiences)
            else:
                # Target experience is excluded from the experiences yielded to the listener:
                listener_experiences = self.sample(idx=None, excepts=[idx])[0].unsqueeze(0)
                # The target_decision_idx is set to `nbr_experiences`:
                target_decision_idx = (self.kwargs['nbr_distractors']+1)*torch.ones(1).long()                         
        else:
            experiences = self.sample(idx=idx)[0].unsqueeze(0)
            listener_experiences, target_decision_idx = shuffle(experiences)
            speaker_experiences = experiences 
            if self.kwargs['observability'] == "partial":
                speaker_experiences = speaker_experiences[:,0].unsqueeze(1)

        output_dict = {'speaker_experiences':speaker_experiences,
                       'listener_experiences':listener_experiences,
                       'target_decision_idx':target_decision_idx}

        return output_dict

        
