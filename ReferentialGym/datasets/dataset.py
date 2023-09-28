from typing import Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset as torchDataset
import numpy as np 
import copy

import wandb 


DC_version = 1 
OC_version = 1 
DSS_version = 1
"""
DC_version ==2 implies that the batch size is split between
examples where the target is retained, and examples where the target
is made different.

Need to find out how does this affect the descriptive ratio sampling?

The batch size is artificially regularised to fit the user params, still.

Upon retaining the target stimuli, they are still resampled in order
to benefit from egocentrism, for instance.

DSS_version == 2 implies that the Distractors Sampling Scheme is relying
on a per-target distractor sampling likelihood confusion matrix that can be 
updated by the user in an online fashion.
"""

from ReferentialGym.datasets.utils import unsqueeze, concatenate


def shuffle(experiences, orders=None):
    st_size = experiences.shape
    batch_size = st_size[0]
    nbr_distractors_po = st_size[1]
    if nbr_distractors_po == 1:
        shuffled_experiences = experiences
        perms = [torch.randperm(1).unsqueeze(0)]*batch_size
        perms = concatenate(perms, dim=0)
        decision_target = (perms==0).max(dim=1)[1].long()
        output_order = [0]*batch_size
        return shuffled_experiences, decision_target, output_order
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
        shuffled_experiences.append( unsqueeze(experiences[b,perm,...], 0))
    perms = concatenate(perms,  dim=0)
    shuffled_experiences = concatenate(shuffled_experiences,  dim=0)
    decision_target = (perms==0).max(dim=1)[1].long()
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

    def getNbrDistractors(self, mode='train'):
        return self.nbr_distractors[mode]

    def setNbrDistractors(self, nbr_distractors, mode='train'):
        assert(nbr_distractors > 0)
        self.nbr_distractors[mode] = nbr_distractors

    def __len__(self) -> int:
        raise NotImplementedError

    def getNbrClasses(self) -> int:
        raise NotImplementedError

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
        :param excepts: None, or List of indices (Integers) that are not considered for sampling.
        :param excepts_class: None, or List of keys (Strings or Integers) that corresponds to entries in self.classes.
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
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Samples target experience and distractor experiences according to the distractor sampling scheme.
        
        TODO: In object_centric/class_centric mode, if some distractors are sampled, then it is important to sample
        them from a list of indices that exclude the elements of the class the target belongs to.
        So far, object_centric mode is only used without distractors.

        :params idx: int, index of the experiences to use as a target.
        :returns:
            - `output_dict`: Dict of elements: 
                - 'speaker_experiences': Tensor of shape `(1, A, nbr_stimulus, stimulus_shape)` where `A=nbr_distractors+1` 
                                      if `self.kwargs['observability']=='full'` or `A=1` otherwise (`'partial'`).
                - some other relevant values aligned with the speaker experiences, the keys starts with "speaker_".
                - 'listener_experiences': Tensor of shape `(1, nbr_distractors+1, nbr_stimulus, stimulus_shape)`.
                - some other relevant values aligned with the listener experiences, the keys starts with "listener_".
                - 'target_decision_idx': Tensor of type Long and shape `(1,)` containing the index of the target experience
                                         among the 'listener_experiences'.
        '''

        from_class = None
        if 'similarity' in self.kwargs['distractor_sampling']:
            similarity_ratio = float(self.kwargs['distractor_sampling'].split('-')[-1])/100.0
            sampled_d = self.sample(idx=idx, target_only=True)
            exp_labels = sampled_d["exp_labels"]
            #_, _, exp_labels, _ = self.sample(idx=idx, target_only=True)
            from_class = None
            rv = torch.rand(size=(1,)).item()  
            wandb.log({
                "Dataset/similarity_ratio": similarity_ratio,
                "Dataset/random": rv,
                },
                commit=False,
            )
            if rv < similarity_ratio:
                from_class = exp_labels

        sample_d = self.sample(idx=idx, from_class=from_class)
        exp_labels = sample_d["exp_labels"]
        
        # Adding batch dimension:
        for k,v in sample_d.items():
            #if not(isinstance(v, torch.Tensor)):    
            #    v = torch.Tensor(v)
            if isinstance(v, torch.Tensor): v = v.unsqueeze(0)
            else:   v = np.array(v)[np.newaxis, ...]
            sample_d[k] = v#.unsqueeze(0)

         

        ##--------------------------------------------------------------
        ##--------------------------------------------------------------
        global DC_version

        # Creating listener's dictionnary:
        listener_sample_d = copy.deepcopy(sample_d)
                
        retain_target = True
        if DC_version == 1 \
        and self.kwargs["descriptive"]:
            retain_target = torch.rand(size=(1,)).item() < self.kwargs['descriptive_target_ratio']
            # Target experience is excluded from the experiences yielded to the listener:
            if not retain_target:
                # Sample a new element for the listener to consider.
                # Different from the target element in itself, but also in its class:
                new_target_for_listener_sample_d = self.sample(
                    idx=None, 
                    from_class=from_class, 
                    target_only=True, 
                    excepts=[idx], 
                    excepts_class=[exp_labels[0]] if self.kwargs['object_centric'] else [],
                )
                # Adding batch dimension:
                for k,v in new_target_for_listener_sample_d.items():
                    #if not(isinstance(v, torch.Tensor)):    
                    #    v = torch.Tensor(v)
                    if not isinstance(v, dict): v = torch.Tensor(v).unsqueeze(0)
                    else:   v = np.array(v)[np.newaxis, ...]
                    listener_sample_d[k][:,0] = v#.unsqueeze(0)
        elif DC_version ==2 \
        and self.kwargs["descriptive"]:
            """
            listener_sample_d is the one with retain target,
            and we create a diff_listener_sample_d which does not ever retain target,
            especially not the target index, and possibly not the target class either
            when object-centrism is in effect:
            """
            diff_listener_sample_d = copy.deepcopy(sample_d)
            # Sample a new element for the listener to consider.
            # Different from the target element in itself, but also in its class:
            new_target_for_listener_sample_d = self.sample(
                idx=None, 
                from_class=from_class, 
                target_only=True, 
                excepts=[idx], 
                excepts_class=[exp_labels[0]] if self.kwargs['object_centric'] else [],
            )
            # Adding batch dimension:
            for k,v in new_target_for_listener_sample_d.items():
                #if not(isinstance(v, torch.Tensor)):    
                #    v = torch.Tensor(v)
                #if isinstance(v, list): v = concatenate(v, dim=0)
                if isinstance(v, np.ndarray): v = torch.from_numpy(v)
                #if not isinstance(v, dict): v = torch.Tensor(v).unsqueeze(0)
                #else:   v = np.array(v)[np.newaxis, ...]
                diff_listener_sample_d[k][:,0] = unsqueeze(v, 0)
            
             

        # Object-Centric or Stimulus-Centric?
        global OC_version
        if OC_version == 1 \
        and retain_target and self.kwargs['object_centric']:
            new_target_for_listener_sample_d = self.sample(
                idx=None, 
                from_class=[exp_labels[0]],
                excepts=[idx],  # Make sure to not sample the actual target!
                target_only=True
            )
            # Adding batch dimension:
            for k,v in new_target_for_listener_sample_d.items():
                #if not(isinstance(v, torch.Tensor)):    
                #    v = torch.Tensor(v)
                if not isinstance(v, dict): v = torch.Tensor(v).unsqueeze(0)
                else:   v = np.array(v)[np.newaxis, ...]
                listener_sample_d[k][:,0] = v#.unsqueeze(0)
        elif OC_version == 2 \
        and retain_target:
            """
            Independently of OC, we need to resample in order to benefit from egocentrism:
            So, if OC is in effect, we sample a new target stimulus from the target class,
            with the exception of the actual target stimulus presented to the speaker ;
            and if OC is not in effect then we resample the very same target stimulus, 
            by carrying forward its index:
            """
            new_target_for_listener_sample_d = self.sample(
                idx=None if self.kwargs['object_centric'] else idx, 
                from_class=[exp_labels[0]],
                excepts=[idx] if self.kwargs['object_centric'] else None,  # Make sure to not sample the actual target!
                target_only=True
            )
            # Adding batch dimension:
            for k,v in new_target_for_listener_sample_d.items():
                #if not(isinstance(v, torch.Tensor)):    
                #    v = torch.Tensor(v)
                #if isinstance(v, torch.Tensor): v = v.unsqueeze(0)
                #else:   v = np.array(v)[np.newaxis, ...]
                #listener_sample_d[k][:,0] = v#.unsqueeze(0)
                listener_sample_d[k][:,0]= unsqueeze(v, 0)

        listener_sample_d["experiences"], target_decision_idx, orders = shuffle(listener_sample_d["experiences"])
        if not retain_target:   
            # The target_decision_idx is set to `nbr_experiences`:
            target_decision_idx = (self.nbr_distractors[self.mode]+1)*torch.ones(1).long()
        if DC_version == 2 \
        and self.kwargs['descriptive']:
            diff_target_decision_idx = (self.nbr_distractors[self.mode]+1)*torch.ones(1).long()

        if self.nbr_distractors[self.mode] > 0 :
            for k,v in listener_sample_d.items():
                if k == "experiences":  continue
                listener_sample_d[k], _, _ = shuffle(v, orders=orders)
                if DC_version == 2 \
                and self.kwargs['descriptive']:
                    diff_listener_sample_d[k], _, _ = shuffle(diff_listener_sample_d[k], orders=orders)

        ##--------------------------------------------------------------
        ##--------------------------------------------------------------

        # Creating speaker's dictionnary:
        speaker_sample_d = copy.deepcopy(sample_d)
        if self.kwargs['observability'] == "partial":
            for k,v in speaker_sample_d.items():
                speaker_sample_d[k] = unsqueeze(v[:,0], 1)
        
        if DC_version == 2 \
        and self.kwargs['descriptive']:
            """
            duplicating speaker dictionnary, and
            collating everything together from listener dicts...
            """
            same_speaker_sample_d = copy.deepcopy(sample_d)
            resampled_speaker_sample_d = self.sample(
                idx=None if self.kwargs['object_centric'] else idx, 
                from_class=[exp_labels[0]],
                excepts=[idx] if self.kwargs['object_centric'] else None,  # Make sure to not sample the actual target!
                target_only=True
            )
            if self.kwargs['observability'] == "partial":
                for k,v in same_speaker_sample_d.items():
                    same_speaker_sample_d[k] = unsqueeze(v[:,0], 1)
             # Adding batch dimension:
            for k,v in resampled_speaker_sample_d.items():
                #if not(isinstance(v, torch.Tensor)):    
                #    v = torch.Tensor(v)
                if not isinstance(v, dict): v = torch.Tensor(v).unsqueeze(0)
                else:   v = np.array(v)[np.newaxis, ...]
                same_speaker_sample_d[k][:,0] = v#.unsqueeze(0)
            
            #TODO: check whether the following is necessary or not:
            # it seems it is necessary to make sure we do not have 50%accuracy as target:
            add_extra = torch.rand(size=(1,)).item() < (1.0-self.kwargs['descriptive_target_ratio'])
            if add_extra:
                for k,v in speaker_sample_d.items():
                    speaker_sample_d[k] = concatenate(
                        [v,same_speaker_sample_d[k]], 
                         dim=0,
                    )
                for k,v in listener_sample_d.items():
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    if isinstance(diff_listener_sample_d[k], np.ndarray):
                        diff_listener_sample_d[k] = torch.from_numpy(diff_listener_sample_d[k])
                    listener_sample_d[k] = concatenate(
                        [v, diff_listener_sample_d[k]],
                         dim=0,
                    )
                target_decision_idx = concatenate([target_decision_idx, diff_target_decision_idx],  dim=0)

        #output_dict = {"target_decision_idx":target_decision_idx.long()}
        output_dict = {"target_decision_idx":target_decision_idx}
        for k,v in listener_sample_d.items():
            if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            output_dict[f"listener_{k}"] = v.float() if isinstance(v, torch.Tensor) else v.astype(np.float32)
        for k,v in speaker_sample_d.items():
            if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            output_dict[f"speaker_{k}"] = v.float() if isinstance(v, torch.Tensor) else v.astype(np.float32)
            #output_dict[f"speaker_{k}"] = v.float() 
        
        return output_dict
