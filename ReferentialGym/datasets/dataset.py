from typing import Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset as torchDataset
import numpy as np 

def shuffle(experiences, orders=None):
    st_size = experiences.shape
    batch_size = st_size[0]
    nbr_distractors_po = st_size[1]
    perms = []
    shuffled_experiences = []
    output_order = []
    for b in range(batch_size):
        if orders is None:
            perm = torch.randperm(nbr_distractors_po)
        else: 
            perm = orders[b]
        #if experiences.is_cuda: perm = perm.cuda()
        output_order.append(perm)
        perms.append(perm.unsqueeze(0))
        shuffled_experiences.append( experiences[b,perm,...].unsqueeze(0))
    perms = torch.cat(perms, dim=0)
    shuffled_experiences = torch.cat(shuffled_experiences, dim=0)
    decision_target = (perms==0).max(dim=1)[1].long()
    return shuffled_experiences, decision_target, output_order

def np_shuffle(experiences, orders=None):
    st_size = experiences.shape
    batch_size = st_size[0]
    nbr_distractors_po = st_size[1]
    perms = []
    shuffled_experiences = []
    output_order = []
    for b in range(batch_size):
        if orders is None:
            perm = np.random.permutation(nbr_distractors_po)
        else: 
            perm = orders[b]
        output_order.append(perm)
        perms.append(np.expand_dims(perm, 0))
        shuffled_experiences.append( np.expand_dims(experiences[b,perm,...], 0))
    perms = np.concatenate(perms, axis=0)
    shuffled_experiences = np.concatenate(shuffled_experiences, axis=0)
    decision_target = (perms==0).max(axis=1).astype('int64')
    return shuffled_experiences, decision_target, output_order

class Dataset(torchDataset):
    def __init__(self, kwargs):
        '''
        :attribute classes: List (or Dictionary) of Lists of (absolute) indices of experiences.
        '''
        super(Dataset,self).__init__()

        self.kwargs = kwargs
        if "root_folder" in kwargs:
            self.root_folder = kwargs['root_folder']
        
        self.nbr_distractors = self.kwargs['nbr_distractors']
        self.nbr_stimulus = self.kwargs['nbr_stimulus']
        
        self.classes = None 

    def getNbrDistractors(self):
        return self.nbr_distractors

    def setNbrDistractors(self, nbr_distractors):
        assert(nbr_distractors > 0)
        self.nbr_distractors = nbr_distractors

    def __len__(self) -> int:
        raise NotImplementedError

    def getNbrClasses(self) -> int:
        raise NotImplementedError

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
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        from_class = None
        if 'similarity' in self.kwargs['distractor_sampling']:
            similarity_ratio = float(self.kwargs['distractor_sampling'].split('-')[-1])
            _, _, exp_labels, _ = self.sample(idx=idx, target_only=True)
            from_class = None
            if torch.rand(size=(1,)).item() < similarity_ratio:
                from_class = exp_labels

        if self.kwargs['descriptive']:
            experiences, indices, exp_labels, exp_latents = self.sample(idx=idx, from_class=from_class)
            experiences = experiences.unsqueeze(0)
            latent_experiences = exp_latents.unsqueeze(0)
            # In descriptive mode, the speaker observability is always partial:
            speaker_experiences = experiences[:,0].unsqueeze(1)
            speaker_latent_experiences = latent_experiences[:,0].unsqueeze(1)
            
            if torch.rand(size=(1,)).item() < self.kwargs['descriptive_target_ratio']:
                # Target experience remain in the experiences yielded to the listener:
                listener_experiences = experiences 
                listener_latent_experiences = latent_experiences 
                if self.kwargs['object_centric']:
                    lexp, _, _, lexp_latent = self.sample(idx=None, from_class=[exp_labels[0]], target_only=True)
                    listener_experiences[:,0] = lexp.unsqueeze(0)
                    listener_latent_experiences[:,0] = lexp_latent.unsqueeze(0)
                listener_experiences, target_decision_idx, orders = shuffle(listener_experiences)
                listener_latent_experiences, _, _ = shuffle(listener_latent_experiences, orders=orders)
            else:
                # Target experience is excluded from the experiences yielded to the listener:
                lexp, _, _, lexp_latent = self.sample(idx=None, from_class=from_class, excepts=[idx])
                listener_experiences = lexp.unsqueeze(0)
                listener_latent_experiences = lexp_latent.unsqueeze(0)
                #listener_experiences = self.sample(idx=None, from_class=from_class, excepts=[idx])[0].unsqueeze(0)
                # The target_decision_idx is set to `nbr_experiences`:
                target_decision_idx = (self.nbr_distractors+1)*torch.ones(1).long()
        else:
            experiences, indices, exp_labels, exp_latents = self.sample(idx=idx, from_class=from_class)
            experiences = experiences.unsqueeze(0)
            exp_latents = exp_latents.unsqueeze(0)
            
            listener_experiences, target_decision_idx, orders = shuffle(experiences)
            listener_latent_experiences, _, _ = shuffle(exp_latents, orders=orders)
            
            speaker_experiences = experiences 
            speaker_latent_experiences = exp_latents
            
            if self.kwargs['observability'] == "partial":
                speaker_experiences = speaker_experiences[:,0].unsqueeze(1)
                speaker_latent_experiences = speaker_latent_experiences[:,0].unsqueeze(1)
        
        output_dict = {'speaker_experiences':speaker_experiences,
                       'listener_experiences':listener_experiences,
                       'speaker_latent_experiences':speaker_latent_experiences,
                       'listener_latent_experiences':listener_latent_experiences,
                       'target_decision_idx':target_decision_idx}

        return output_dict


    def __npgetitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        from_class = None
        if 'similarity' in self.kwargs['distractor_sampling']:
            similarity_ratio = float(self.kwargs['distractor_sampling'].split('-')[-1])
            _, _, exp_labels, _ = self.sample(idx=idx, target_only=True)
            from_class = None
            if torch.rand(size=(1,)).item() < similarity_ratio:
                from_class = exp_labels

        if self.kwargs['descriptive']:
            experiences, indices, exp_labels, exp_latents = self.sample(idx=idx, from_class=from_class)
            experiences = np.expand_dims(experiences, axis=0)
            latent_experiences = np.expand_dims(exp_latents, 0)
            # In descriptive mode, the speaker observability is always partial:
            speaker_experiences = np.expand_dims(experiences[:,0], 1)
            speaker_latent_experiences = np.expand_dims(latent_experiences[:,0], 1)

            if torch.rand(size=(1,)).item() < self.kwargs['descriptive_target_ratio']:
                # Target experience remain in the experiences yielded to the listener:
                listener_experiences = experiences 
                listener_latent_experiences = latent_experiences 
                if self.kwargs['object_centric']:
                    lexp, _, _, lexp_latent = self.sample(idx=None, from_class=[exp_labels[0]], target_only=True)
                    listener_experiences[:,0] = np.expand_dims(lexp, 0)
                    listener_latent_experiences[:,0] = np.expand_dims(lexp_latent, 0)
                listener_experiences, target_decision_idx, orders = np_shuffle(listener_experiences)
                listener_latent_experiences, _, _ = np_shuffle(listener_latent_experiences, orders=orders)
            else:
                # Target experience is excluded from the experiences yielded to the listener:
                lexp, _, _, lexp_latent = self.sample(idx=None, from_class=from_class, excepts=[idx])
                listener_experiences = np.expand_dims(lexp, 0)
                listener_latent_experiences = np.expand_dims(lexp_latent, 0)
                # The target_decision_idx is set to `nbr_experiences`:
                target_decision_idx = (self.nbr_distractors+1)*np.ones(1).long()
        else:
            experiences, indices, exp_labels, exp_latents = self.sample(idx=idx, from_class=from_class)
            experiences = np.expand_dims(experiences, 0)
            exp_latents = np.expand_dims(exp_latents, 0)

            listener_experiences, target_decision_idx, orders = np_shuffle(experiences)
            listener_latent_experiences, _, _ = np_shuffle(exp_latents, orders=orders)
            
            speaker_experiences = experiences 
            speaker_latent_experiences = exp_latents
            
            if self.kwargs['observability'] == "partial":
                speaker_experiences = np.expand_dims(speaker_experiences[:,0], 1)
                speaker_latent_experiences = np.expand_dims(speaker_latent_experiences[:,0], 1)

        '''
        speaker_experiences = torch.from_numpy(speaker_experiences)
        listener_experiences = torch.from_numpy(listener_experiences)
        speaker_latent_experiences = torch.from_numpy(speaker_latent_experiences)
        listener_latent_experiences = torch.from_numpy(listener_latent_experiences)
        target_decision_idx = torch.from_numpy(target_decision_idx)
        '''

        output_dict = {'speaker_experiences':speaker_experiences,
                       'listener_experiences':listener_experiences,
                       'speaker_latent_experiences':speaker_latent_experiences,
                       'listener_latent_experiences':listener_latent_experiences,
                       'target_decision_idx':target_decision_idx}

        return output_dict
