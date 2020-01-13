import os 
import torch
import torchvision
from torchvision.datasets import CocoDetection
from PIL import Image 
import numpy as np

VERBOSE = False

class MSCOCODataset(CocoDetection):
    """
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, root, annFile, transform=None):
        super(MSCOCODataset, self).__init__(root=root, annFile=annFile, transform=transform)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cats_names = [cat['name'] for cat in self.cats]
        
        self.cats_idx = {}
        for idx, cat in enumerate(self.cats):
            self.cats_idx[cat['id']] = idx
        
        self.latent_values = []
        self.targets = []
        nbr_removed_imgs = 0
        idx = 0
        while idx < len(self):
            img_id = self.ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            if len(target):
                self.latent_values.append(np.zeros(len(self.cats_names)))
                for bb in target:
                    self.latent_values[-1][self.cats_idx[bb['category_id']]] = 1 
                # The target is simply the first bounding box' category id...
                self.targets.append(self.cats_idx[target[0]['category_id']])
                idx += 1
            else:
                del self.ids[idx]
                nbr_removed_imgs += 1
                if VERBOSE: print(f'WARNING: removing Image ID {img_id}...')

        print('Dataset loaded : OK.')
        print(f'Nbr removed image: {nbr_removed_imgs}.')

    def __len__(self):
        return len(self.ids)

    def getclass(self, idx):
        return self.targets[idx]

    def getlatentvalue(self, idx):
        return self.latent_values[idx]
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        target = self.getclass(index)
        latent_value = torch.from_numpy(self.getlatentvalue(index))
        
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, latent_value
