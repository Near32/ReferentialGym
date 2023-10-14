from typing import Dict, List, Tuple
import torch
from torch.utils.data.dataset import Dataset as torchDataset
import numpy as np 
import copy

import wandb 


DC_version = 1 
OC_version = 1 
"""
DC_version ==2 implies that the batch size is split between
examples where the target is retained, and examples where the target
is made different.

Need to find out how does this affect the descriptive ratio sampling?

The batch size is artificially regularised to fit the user params, still.

Upon retaining the target stimuli, they are still resampled in order
to benefit from egocentrism, for instance.

"""

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
        self.original_object_centric_type = self.kwargs.get('object_centric_type', 'hard') 
    
    def getNbrDistractors(self, mode='train'):
        return self.nbr_distractors[mode]

    def setNbrDistractors(self, nbr_distractors, mode='train'):
        assert(nbr_distractors > 0)
        self.nbr_distractors[mode] = nbr_distractors

    def size(self) -> int:
        raise NotImplementedError
    
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
            if not(isinstance(v, torch.Tensor)):    
                v = torch.Tensor(v)
            sample_d[k] = v.unsqueeze(0)

         

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
                    if not(isinstance(v, torch.Tensor)):    
                        v = torch.Tensor(v)
                    listener_sample_d[k][:,0] = v.unsqueeze(0)
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
                if not(isinstance(v, torch.Tensor)):    
                    v = torch.Tensor(v)
                diff_listener_sample_d[k][:,0] = v.unsqueeze(0)
            
             

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
                if not(isinstance(v, torch.Tensor)):    
                    v = torch.Tensor(v)
                listener_sample_d[k][:,0] = v.unsqueeze(0)
        elif OC_version == 2 \
        and retain_target:
            """
            Independently of OC, we need to resample in order to benefit from egocentrism:
            So, if OC is in effect, we sample a new target stimulus from the target class,
            with the exception of the actual target stimulus presented to the speaker ;
            and if OC is not in effect then we resample the very same target stimulus, 
            by carrying forward its index:
            """
            # Default: Hard-OC : Make sure to not sample the actual target!
            excepts = [idx] 
            new_idx = None
            
            if self.kwargs['object_centric'] \
            and 'ratio' in self.original_object_centric_type:
                percentage = float(self.original_object_centric_type.split('-')[-1])
                OC_rv = torch.rand(size=(1,)).item()
                hard_OC = OC_rv < percentage/100.0
                if hard_OC:
                    self.kwargs['object_centric_type'] = 'hard'
                else:
                    self.kwargs['object_centric_type'] = 'extra-simple'
                wandb.log({
                  "Dataset/OC_ratio": percentage/100.0,
                  "Dataset/OC_random": OC_rv,
                  },
                  commit=False,
                )
             
            if self.kwargs['object_centric'] \
            and self.kwargs.get('object_centric_type', 'hard')=='extra-simple':
                excepts = None
                new_idx = idx
            elif self.kwargs['object_centric'] \
            and self.kwargs.get('object_centric_type', 'hard')=='simple':
                excepts = None 
                new_idx = None
            
            new_target_for_listener_sample_d = self.sample(
                idx=new_idx, 
                from_class=[exp_labels[0]],
                excepts=excepts,
                target_only=True
            )
             
            new_target_for_listener_sample_d = self.sample(
                idx=new_idx, 
                from_class=[exp_labels[0]],
                excepts=excepts,
                target_only=True
            )
            # Adding batch dimension:
            for k,v in new_target_for_listener_sample_d.items():
                if not(isinstance(v, torch.Tensor)):    
                    v = torch.Tensor(v)
                listener_sample_d[k][:,0] = v.unsqueeze(0)

        listener_sample_d["experiences"], target_decision_idx, orders = shuffle(listener_sample_d["experiences"])
        if not retain_target:   
            # The target_decision_idx is set to `nbr_experiences`:
            target_decision_idx = (self.nbr_distractors[self.mode]+1)*torch.ones(1).long()
        if DC_version == 2 \
        and self.kwargs['descriptive']:
            diff_target_decision_idx = (self.nbr_distractors[self.mode]+1)*torch.ones(1).long()

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
                speaker_sample_d[k] = v[:,0].unsqueeze(1)
        
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
                    same_speaker_sample_d[k] = v[:,0].unsqueeze(1)
             # Adding batch dimension:
            for k,v in resampled_speaker_sample_d.items():
                if not(isinstance(v, torch.Tensor)):    
                    v = torch.Tensor(v)
                same_speaker_sample_d[k][:,0] = v.unsqueeze(0)
            
            add_extra = torch.rand(size=(1,)).item() < (1.0-self.kwargs['descriptive_target_ratio'])
            if add_extra:
                for k,v in speaker_sample_d.items():
                    speaker_sample_d[k] = torch.cat(
                        [v,same_speaker_sample_d[k]], 
                        dim=0,
                    )
                for k,v in listener_sample_d.items():
                    listener_sample_d[k] = torch.cat(
                        [v, diff_listener_sample_d[k]],
                        dim=0,
                    )
                target_decision_idx = torch.cat([target_decision_idx, diff_target_decision_idx], dim=0)

        output_dict = {"target_decision_idx":target_decision_idx}
        for k,v in listener_sample_d.items():
            output_dict[f"listener_{k}"] = v
        for k,v in speaker_sample_d.items():
            output_dict[f"speaker_{k}"] = v 
        
        return output_dict
