from typing import Dict, List, Tuple
import torch 
from torch.utils.data import Dataset 
import os
import pickle 
import numpy as np
import random
import cv2
from PIL import Image 


def generate_dataset(root,
                     train_size=9800,
                     test_size=200,
                     img_size=75,
                     object_size=5,
                     nb_objects=6,
                     nb_questions=10
                    ):
    '''
    Adapted from:

    BSD 3-Clause License
    Copyright (c) 2017, Kim Heecheol All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
    OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    '''
    
    size = object_size
    '''
    question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
    """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
    '''
    question_size = nb_objects+5 
    ## nb_objects(==nb_colors) for one-hot vector of color, 2 for question type, 3 for question subtype
    """Answer : [yes, no, rectangle, circle, *colors]"""
    
    dirs = root 
    '''
    colors = [
        (0,0,255),##r
        (0,255,0),##g
        (255,0,0),##b
        (0,156,255),##o
        (128,128,128),##k
        (0,255,255)##y
    ]
    '''

    # Reproducing: 
    # http://alumni.media.mit.edu/~wad/color/numbers.html      
    # without white...
    colors = [
    #Black
    (0, 0, 0),
    #Dk. Gray
    (87, 87, 87), 
    #Red
    (173, 35, 35), 
    #Blue
    (42, 75, 215), 
    #Green
    (29, 105, 20), 
    #Brown
    (129, 74, 25), 
    #Purple
    (129, 38, 192), 
    #Lt. Gray
    (160, 160, 160), 
    #Lt. Green
    (129, 197, 122), 
    #Lt. Blue
    (157, 175, 255), 
    #Cyan
    (41, 208, 208), 
    #Orange
    (255, 146, 51), 
    #Yellow
    (255, 238, 51), 
    #Tan
    (233, 222, 187), 
    #Pink
    (55, 205, 243), 
    ]

    assert(nb_objects<=len(colors))
    colors = colors[:nb_objects]

    shapes = [
        "circle",
        "rectangle"
    ]
    
    # 0, as a class, is a lack of object (no color/ no shape):
    latent_one_hot_repr_sizes = {"color":len(colors)+1,
        "shape":len(shapes)+1,
    }

    size_one_hot_vec_per_object = sum([v for k,v in latent_one_hot_repr_sizes.items()])
    nb_attr_per_object = len(latent_one_hot_repr_sizes)

    pos_side = np.arange(object_size, img_size-object_size+1, 2*object_size)
        
    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))

    def find_pos_side_bucket(coord, pos_side):
        '''
        bucket = 0
        while coord <= pos_side[bucket]+object_size:
            bucket += 1
            if bucket >= len(pos_side):
                raise AssertionError
        return bucket-1
        '''
        return max(0, coord-1) // (2*object_size)

    def center_generate(objects):
        while True:
            pas = True
            center = np.random.randint(0+size, img_size - size, 2)        
            if len(objects) > 0:
                for obj in objects:
                    name,c,shape = obj[:3]
                    if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                        pas = False
            if pas:
                return center



    def build_dataset():
        objects = []
        latent_value = np.stack([np.zeros(size_one_hot_vec_per_object) for _ in range(len(pos_side)**2)]).reshape((len(pos_side),len(pos_side),-1))
        # color: blank
        latent_value[:,:,0] = 1
        # sahpe: blank
        latent_value[:,:,nb_objects+1] = 1
        latent_class = np.stack([np.zeros(nb_attr_per_object) for _ in range(len(pos_side)**2)]).reshape((len(pos_side),len(pos_side),-1))

        img = np.ones((img_size,img_size,3)) * 255
        for color_id,color in enumerate(colors[:nb_objects]):  
            center = center_generate(objects)
            bx = find_pos_side_bucket(center[0], pos_side)
            by = find_pos_side_bucket(center[1], pos_side)
            if random.random()<0.5:
                start = (center[0]-size, center[1]-size)
                end = (center[0]+size, center[1]+size)
                cv2.rectangle(img, start, end, color, -1)
                objects.append((color_id,center,'r',bx,by))
            else:
                center_ = (center[0], center[1])
                cv2.circle(img, center_, size, color, -1)
                objects.append((color_id,center,'c',bx,by))

        '''
        '''
        for obj_id in range(len(objects)):
            bx = objects[obj_id][3]
            by = objects[obj_id][4]
            
            color_repr = np.zeros(latent_one_hot_repr_sizes["color"])
            # 0, as a class is lack of color/shape:
            color_repr[objects[obj_id][0]+1] = 1

            latent_class[by][bx][0] = objects[obj_id][0]+1
            
            shape_repr = np.zeros(latent_one_hot_repr_sizes["shape"])
            shape_id = 0 if objects[obj_id][2]=='c' else 1
            shape_repr[shape_id+1] = 1
            
            latent_class[by][bx][1] = shape_id
            
            latent_value[by][bx] = np.concatenate([color_repr, shape_repr], axis=0)
        
        '''
        '''

        rel_questions = []
        norel_questions = []
        rel_answers = []
        norel_answers = []
        """Non-relational questions"""
        for _ in range(nb_questions):
            question = np.zeros((question_size))
            color = random.randint(0,nb_objects-1)
            question[color] = 1
            question[nb_objects] = 1
            subtype = random.randint(0,2)
            question[subtype+2+nb_objects] = 1
            norel_questions.append(question)
            """Answer : [yes, no, rectangle, circle, *colors]"""
            if subtype == 0:
                """query shape->rectangle/circle"""
                if objects[color][2] == 'r':
                    answer = 2
                else:
                    answer = 3

            elif subtype == 1:
                """query horizontal position->yes/no"""
                if objects[color][1][0] < img_size / 2:
                    answer = 0
                else:
                    answer = 1

            elif subtype == 2:
                """query vertical position->yes/no"""
                if objects[color][1][1] < img_size / 2:
                    answer = 0
                else:
                    answer = 1
            norel_answers.append(answer)
        
        """Relational questions"""
        for i in range(nb_questions):
            question = np.zeros((question_size))
            color = random.randint(0,nb_objects-1)
            question[color] = 1
            question[nb_objects+1] = 1
            subtype = random.randint(0,2)
            question[subtype+2+nb_objects] = 1
            rel_questions.append(question)

            if subtype == 0:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = 999
                closest = dist_list.index(min(dist_list))
                if objects[closest][2] == 'r':
                    answer = 2
                else:
                    answer = 3
                    
            elif subtype == 1:
                """furthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                furthest = dist_list.index(max(dist_list))
                if objects[furthest][2] == 'r':
                    answer = 2
                else:
                    answer = 3

            elif subtype == 2:
                """count->1~nb_objects"""
                my_obj = objects[color][2]
                count = -1
                for obj in objects:
                    if obj[2] == my_obj:
                        count +=1 
                answer = count+4

            rel_answers.append(answer)

        relations = (rel_questions, rel_answers)
        norelations = (norel_questions, norel_answers)
        
        img = img/255.
        dataset = (img, relations, norelations, latent_class.reshape(-1), latent_value.reshape(-1))
        return dataset


    print('building test datasets...')
    test_datasets = [build_dataset() for _ in range(test_size)]
    print('building train datasets...')
    train_datasets = [build_dataset() for _ in range(train_size)]


    print('saving datasets...')
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((train_datasets, test_datasets), f)
    print('datasets saved at {}'.format(filename))

    return (train_datasets, test_datasets)


class SortOfCLEVRDataset(Dataset):
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 generate=False, 
                 nbrSampledQstPerImg=1,
                 train_size=9800,
                 test_size=200,
                 img_size=75,
                 object_size=5,
                 nb_objects=6,
                 nb_questions=10
                 ):
        super(SortOfCLEVRDataset, self).__init__()
        
        self.root = root
        self.file = 'sort-of-clevr.pickle'
        self.nbrSampledQstPerImg = nbrSampledQstPerImg

        if not self._check_exists():
            if generate:
                train_datasets, test_datasets = self._generate(root=root,
                                                               train_size=train_size,
                                                               test_size=test_size,
                                                               img_size=img_size,
                                                               object_size=object_size,
                                                               nb_objects=nb_objects,
                                                               nb_questions=nb_questions)
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
        else:
            filepath = os.path.join(self.root, self.file)
            with open(filepath, 'rb') as f:
              train_datasets, test_datasets = pickle.load(f)
            
        self.train = train 
        if self.train:
            datasets = train_datasets
        else:
            datasets = test_datasets

        dataset = []
        nbrImage = 0
        nbrQst = 0 
        for img, relations, norelations, latent_class, latent_value in datasets:
            img = img.transpose((2,0,1))
            nbrImage += 1

            bina = np.power( 2, np.arange(len(relations[0][0])))
            relations = [ ( int( np.sum(bina*qst)), ans) for qst, ans in zip(relations[0],relations[1])]
            
            for _ in range(self.nbrSampledQstPerImg):
                if len(relations) == 0:    break
                idxsample = random.choice(range(len(relations)))
                qst, ans = relations.pop(idxsample)
                dataset.append((img,qst,ans,latent_class,latent_value))
                nbrQst += 1

        print("Sort-of-CLEVR Dataset: {} unique images.".format(nbrImage))
        print("Sort-of-CLEVR Dataset: {} questions asked.".format(nbrQst))

        nbr_unique_ans = max([d[2] for d in dataset])+1
        self.dataset = [ (d[0], d[2]+nbr_unique_ans*d[1], d[3], d[4]) for d in dataset]
        
        self.transform = transform 

    def __len__(self) -> int:
        return len(self.dataset)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,self.file))

    def _generate(self, 
                  root,
                  train_size,
                  test_size,
                  img_size,
                  object_size,
                  nb_objects,
                  nb_questions):
        """
        Generate the Sort-of-CLEVR dataset if it doesn't exist already.
        """
        if root is None:
            root = self.root
        os.makedirs(root, exist_ok=True)
        return generate_dataset(root=root,
            train_size=train_size,
            test_size=test_size,
            img_size=img_size,
            object_size=object_size,
            nb_objects=nb_objects,
            nb_questions=nb_questions
            )

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)

        _, target, _, _ = self.dataset[idx]
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

        img, target, latent_class, latent_value = self.dataset[idx]
        
        latent_class = torch.from_numpy(latent_class)
        latent_value = torch.from_numpy(latent_value)

        img = (img*255).astype('uint8').transpose((2,1,0))
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, latent_class, latent_value
