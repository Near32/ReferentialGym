from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset 
import os
import numpy as np
import random
from PIL import Image 


class dSpritesDataset(Dataset) :
    def __init__(self, root='./', train=True, transform=None, split_strategy=None) :
        '''
        :param split_strategy: str 
            e.g.: 'divider-10-offset-0'
        '''
        self.train = train
        self.root = os.path.join(root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        self.transform = transform
        self.split_strategy = split_strategy


        # Load dataset
        dataset_zip = np.load(self.root, encoding='latin1', allow_pickle=True)
        #data = np.load(root, encoding='latin1')
        #data = torch.from_numpy(data['imgs']).unsqueeze(1).float())
        print('Keys in the dataset:')
        for k in dataset_zip.keys(): print(k)
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.targets = np.zeros(len(self.latents_classes)) #[random.randint(0, 10) for _ in self.imgs]
        for idx, latent_cls in enumerate(self.latents_classes):
            posX = latent_cls[-2]
            posY = latent_cls[-1]
            target = posX*32+posY
            self.targets[idx] = target  
        self.metadata = dataset_zip['metadata'][()]
        
        if self.split_strategy is not None:
            strategy = self.split_strategy.split('-')
            if 'divider' in self.split_strategy and 'offset' in self.split_strategy:
                self.divider = int(strategy[1])
                assert(self.divider>0)
                self.offset = int(strategy[-1])
                assert(self.offset>=0 and self.offset<self.divider)
        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        for idx in range(len(self.imgs)):
            if idx % self.divider == self.offset:
                self.indices.append(idx)

        self.train_ratio = 0.8
        end = int(len(self.indices)*self.train_ratio)
        if self.train:
            self.indices = self.indices[:end]
        else:
            self.indices = self.indices[end:]

        print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
        print(f'Number of classes: {len(set([self.targets[idx] for idx in self.indices]))} / 1024.')

        self.imgs = self.imgs[self.indices]
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        self.targets = self.targets[self.indices]
        del self.metadata

        print('Dataset loaded : OK.')
        
    def __len__(self) :
        return len(self.indices)

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        #idx = self.indices[idx]
        #target = self.latents_classes[idx]
        target = self.targets[idx]
        return target

    def getlatentvalue(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        #idx = self.indices[idx]
        latent_value = self.latents_values[idx]
        return latent_value

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)
        #orig_idx = idx
        #idx = self.indices[idx]

        #img, target = self.dataset[idx]
        image = Image.fromarray((self.imgs[idx]*255).astype('uint8'))
        
        #target = self.getclass(orig_idx)
        #latent_value = torch.from_numpy(self.getlatentvalue(orig_idx))
        
        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target, latent_value