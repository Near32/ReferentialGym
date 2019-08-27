import os
import json
import h5py

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def download_preprocess_dataset(path):
    raise NotImplementedError
    # import subprocess
    # dirpath = os.path.dirname(os.path.abspath(__file__))
    # scriptpath = os.path.join(dirpath, 'download_preprocess_CLEVR.sh')
    # scriptpath = os.path.normpath(scriptpath)

    # arg1 = '{}/CLEVRv1.0/'.format(path).replace(' ', '\\ ')

    # arg2 = arg1
    # arg3 = arg2+'CLEVRv1.0'

    # cmds = subprocess.check_output([scriptpath, arg1, arg2, arg3])


def ToLongTensor(data):
    arr = np.asarray(data, dtype=np.int32)
    tensor = torch.LongTensor(arr)
    return tensor

class Rescale(object) :
    def __init__(self, output_size) :
        assert( isinstance(output_size, (int, tuple) ) )
        self.output_size = output_size

    def __call__(self, sample) :
        image = sample
        h,w = image.shape[:2]
        new_h, new_w = self.output_size
        img = cv2.resize(image, (new_h, new_w) )
        return img

class ToTensor(object) :
    def __call__(self, sample) :
        image = sample
        #swap color axis :
        # numpy : H x W x C
        # torch : C x H x W
        image = image.transpose( (2,0,1) )
        sample =  torch.from_numpy(image)
        return sample

default_image_size = 84
default_transform = transforms.Compose([Rescale( (default_image_size,default_image_size) ),
                                       ToTensor()
                                       ])

class CLEVRDataset(Dataset):
    def __init__(self, root, train=True, transform=default_transform, download=False):
        super(CLEVRDataset, self).__init__()

        self.dataset_path = root
        # default='../DATASETS/CLEVR_v1.0/'
        if './' in self.dataset_path:
            self.dataset_path = os.path.join(os.getcwd(), self.dataset_path) 
        

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.train = train
        if self.train:
            self.data_path = os.path.join(self.dataset_path, 'train_questions.h5') 
            self.vocab_path = os.path.join(self.dataset_path, 'vocab.json') 
            self.image_path = os.path.join(self.dataset_path, 'train_questions.h5.paths.npz')
        else:
            self.data_path = os.path.join(self.dataset_path, 'val_questions.h5') 
            self.vocab_path = os.path.join(self.dataset_path, 'vocab.json') 
            self.image_path = os.path.join(self.dataset_path, 'val_questions.h5.paths.npz')

        self.data = h5py.File(self.data_path, 'r')

        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f)

        self.image_path_data = np.load(self.image_path)

        self.idx2question = ToLongTensor(self.data['questions'])
        self.idx2answer = ToLongTensor(self.data['answers'])
        self.idx2imageidx = ToLongTensor(self.data['image_idxs'])
        self.idx2question_family = np.asarray(self.data['question_families'])
        self.image_idx2path = self.image_path_data['paths']    

        dataset = []
        nbrImage = 0
        nbrQst = 0
        for qst_idx in range(len(self.idx2question)):
            qst = self.idx2question_family[qst_idx]
            ans = self.idx2answer[qst_idx].item()
            img_path = os.path.join(self.dataset_path, self.image_idx2path[qst_idx])

            dataset.append((img_path, qst, ans))
            nbrImage += 1
            nbrQst += 1

        nbr_unique_ans = max([d[2] for d in dataset])+1
        #self.dataset = [ (d[0], d[2]+nbr_unique_ans*d[1]) for d in dataset]
        self.dataset = [ (d[0], d[1]) for d in dataset]

        self.transform = transform 

    def __len__(self) -> int:
        return len(self.dataset)

    def _check_exists(self):
        return os.path.exists(self.dataset_path) and os.path.exists(os.path.join(self.dataset_path, 'train_questions.h5'))

    def _download(self):
        """
        Download and preprocess the Sort-of-CLEVR dataset if it doesn't exist already.
        """
        if self._check_exists():
            return
        os.makedirs(self.dataset_path, exist_ok=True)
        download_preprocess_dataset(self.dataset_path)

    def getVocabSize(self):
        return len(self.vocab['question_vocab'])

    def getAnswerVocabSize(self):
        return len(self.vocab['answer_vocab'])

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)

        _, target = self.dataset[idx]
        return target

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)

        img_path, target = self.dataset[idx]

        img = cv2.imread(img_path)
        img = np.asarray(img, dtype=np.float32)
        if self.transform is not None :
            img = self.transform(img)

        return img, target