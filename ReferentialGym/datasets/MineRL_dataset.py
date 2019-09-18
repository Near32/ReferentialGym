from typing import Dict, List, Tuple
import os
import numpy as np
import random
from tqdm import tqdm

import torch

import minerl
from PIL import Image 

from .dataset import Dataset, shuffle


class MineRLDataset(Dataset):
    def __init__(self, kwargs, root, train=True, transform=None, download=False, experiments=['MineRLObtainDiamond-v0'], skip_interval=0):
        super(MineRLDataset, self).__init__(kwargs=kwargs)
        
        self.kwargs = kwargs
        self.root = root
        self.experiments2use = experiments
        self.skip_interval = skip_interval
        
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')


        self.dataset_exp = {exp: minerl.data.make(exp, data_dir=self.root) for exp in self.experiments2use}
        self.dataset_exp_trajname = { exp: self.dataset_exp[exp].get_trajectory_names() for exp in self.experiments2use}
        self.dataset_exp_traj_data = {exp: dict() for exp in self.experiments2use}

        nbrTraj = 50
        nbrTrajTestDivider = 10
        self.dataset_exptraj_data_loaded = dict()
        self.exptraj2int = dict()
        exptrajIdx = 0 
        for exp in self.experiments2use:
            self.dataset_exptraj_data_loaded[exp] = dict()
            dataset_exp_trajname = list()
            for idxtraj, trajname in enumerate(self.dataset_exp_trajname[exp][:nbrTraj]):
                if train:
                    if idxtraj % 2 == 0: continue
                else:
                    if idxtraj % 2 == 1 or idxtraj > nbrTraj//nbrTrajTestDivider: continue

                dataset_exp_trajname.append(trajname)
                self.dataset_exp_traj_data[exp][trajname] = self.dataset_exp[exp].load_data(stream_name=trajname, 
                                                                                            skip_interval=0,
                                                                                            include_metadata=False)
                self.dataset_exptraj_data_loaded[exp+trajname] = True 
                self.exptraj2int[exp+trajname] = exptrajIdx
                exptrajIdx += 1
            self.dataset_exp_trajname[exp] = dataset_exp_trajname

        # Let us define a class as a given trajectory.
        # In doing so, we require the emergence of an artificial language
        # whose degree of expressivity is tailored to distinguish between
        # stimuli coming from the same RL trajectory. That is to say that
        # the artificial language should be viable as an abstraction to
        # each RL trajectory's state sequences. 
        self.trajectories = {}
        self.class2len = {}
        self.dataset = []
        for exp in self.experiments2use:
            print(f'Experiment:: {exp} :: nbr of trajectories:: {len(self.dataset_exp_trajname[exp])}.')
            for trajname in tqdm(self.dataset_exp_trajname[exp]):
                self.trajectories[exp+trajname] = self.dataset_exp_traj_data[exp][trajname]

                exptraj_data = []
                traj = list(self.trajectories[exp+trajname])
                traj_len = len(traj)
                traj_seq_len = traj_len // self.skip_interval
                for idx_traj_seq in range(traj_seq_len):
                    traj_seq = [t[0]['pov'] for t in traj[idx_traj_seq*self.skip_interval:(idx_traj_seq+1)*self.skip_interval]]
                    exptraj_data.append(traj_seq)
                self.trajectories[exp+trajname] = exptraj_data

                self.class2len[exp+trajname] = len(self.trajectories[exp+trajname])
                for idx in range(self.class2len[exp+trajname]):
                    self.dataset.append( (idx, exp+trajname))

        self.classes = {}
        for idx in range(len(self.dataset)):
            _, exptraj = self.dataset[idx]
            cl = self.exptraj2int[exptraj]
            if cl not in self.classes: self.classes[cl] = []
            self.classes[cl].append(idx)

        self.transform = transform 

    def __len__(self) -> int:
        return len(self.dataset)

    def getNbrClasses(self) -> int:
        return len(self.trajectories)
    
    def _check_exists(self, experiments2use=None):
        if experiments2use is None: experiments2use = self.experiments2use
        for exp in experiments2use:
            if not os.path.exists(self.root):
                return False
        return True

    def _download(self):
        """
        Download the MineRL dataset if it doesn't exist already.
        """
        if self._check_exists():
            return
        
        for exp in self.experiments2use:
            if not self._check_exists(experiments2use=[exp]):
                minerl.data.download(self.root, experiment=exp)

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)

        _, target = self.dataset[idx]
        return target

    def _get(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)

        idx_in_exptraj, exptraj = self.dataset[idx]

        imgs, target = self.trajectories[exptraj][idx_in_exptraj], self.exptraj2int[exptraj]
        img = random.choice(imgs)

        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            tr_img = self.transform(img)
        else:
            tr_img = img 

        return tr_img, target

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
        test = True
        not_enough_elements = False
        while test:
            if from_class is None or not_enough_elements:
                from_class = list(self.classes.keys())
                
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
        for idx in indices:
            exp, tc = self._get(idx)
            experiences.append(exp.unsqueeze(0))
            exp_labels.append(tc)
            if target_only: break

        experiences = torch.cat(experiences,dim=0)
        experiences = experiences.unsqueeze(1)
        # account for the temporal dimension...
        
        return experiences, indices, exp_labels

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
        from_class = None
        if 'similarity' in self.kwargs['distractor_sampling']:
            similarity_ratio = float(self.kwargs['distractor_sampling'].split('-')[-1])
            _, _, exp_labels = self.sample(idx=idx, target_only=True)
            from_class = None
            if torch.rand(size=(1,)).item() < similarity_ratio:
                from_class = exp_labels

        if self.kwargs['descriptive']:
            experiences, indices, exp_labels = self.sample(idx=idx, from_class=from_class)
            experiences = experiences.unsqueeze(0)
            # In descriptive mode, the speaker observability is alwasy partial:
            speaker_experiences = experiences[:,0].unsqueeze(1)
            
            if torch.rand(size=(1,)).item() < self.kwargs['descriptive_target_ratio']:
                # Target experience remain in the experiences yielded to the listener:
                listener_experiences = experiences 
                if self.kwargs['object_centric']:
                    '''
                    Using the skip_interval hyperparameter, 
                    we assume that temporal closeness in each trajectories
                    is tantamount to seeing the same object on another viewpoint.
                    Thus, it is an object-centric approach.
                    '''
                    listener_experiences[:,0] = self.sample(idx=idx, from_class=[exp_labels[0]], target_only=True)[0].unsqueeze(0)
                listener_experiences, target_decision_idx = shuffle(listener_experiences)
            else:
                # Target experience is excluded from the experiences yielded to the listener:
                listener_experiences = self.sample(idx=None, from_class=from_class, excepts=[idx])[0].unsqueeze(0)
                # The target_decision_idx is set to `nbr_experiences`:
                target_decision_idx = (self.kwargs['nbr_distractors']+1)*torch.ones(1).long()                         
        else:
            experiences, indices, exp_labels = self.sample(idx=idx, from_class=from_class)
            experiences = experiences.unsqueeze(0)
            listener_experiences, target_decision_idx = shuffle(experiences)
            speaker_experiences = experiences 
            if self.kwargs['observability'] == "partial":
                speaker_experiences = speaker_experiences[:,0].unsqueeze(1)

        output_dict = {'speaker_experiences':speaker_experiences,
                       'listener_experiences':listener_experiences,
                       'target_decision_idx':target_decision_idx}

        return output_dict
