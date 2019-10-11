from typing import Dict, List, Tuple
from torch.utils.data import Dataset 
import os
import pickle 
import numpy as np
import random
import cv2
from PIL import Image 


def generate_dataset(root=None):
    '''
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
    
    train_size = 9800
    test_size = 200
    img_size = 75
    size = 5
    question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
    """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

    nb_questions = 10
    dirs = './datasets/Sort-of-CLEVR/' if root is None else root 

    colors = [
        (0,0,255),##r
        (0,255,0),##g
        (255,0,0),##b
        (0,156,255),##o
        (128,128,128),##k
        (0,255,255)##y
    ]


    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))

    def center_generate(objects):
        while True:
            pas = True
            center = np.random.randint(0+size, img_size - size, 2)        
            if len(objects) > 0:
                for name,c,shape in objects:
                    if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                        pas = False
            if pas:
                return center



    def build_dataset():
        objects = []
        img = np.ones((img_size,img_size,3)) * 255
        for color_id,color in enumerate(colors):  
            center = center_generate(objects)
            if random.random()<0.5:
                start = (center[0]-size, center[1]-size)
                end = (center[0]+size, center[1]+size)
                cv2.rectangle(img, start, end, color, -1)
                objects.append((color_id,center,'r'))
            else:
                center_ = (center[0], center[1])
                cv2.circle(img, center_, size, color, -1)
                objects.append((color_id,center,'c'))


        rel_questions = []
        norel_questions = []
        rel_answers = []
        norel_answers = []
        """Non-relational questions"""
        for _ in range(nb_questions):
            question = np.zeros((question_size))
            color = random.randint(0,5)
            question[color] = 1
            question[6] = 1
            subtype = random.randint(0,2)
            question[subtype+8] = 1
            norel_questions.append(question)
            """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
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
            color = random.randint(0,5)
            question[color] = 1
            question[7] = 1
            subtype = random.randint(0,2)
            question[subtype+8] = 1
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
                """count->1~6"""
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
        dataset = (img, relations, norelations)
        return dataset


    print('building test datasets...')
    test_datasets = [build_dataset() for _ in range(test_size)]
    print('building train datasets...')
    train_datasets = [build_dataset() for _ in range(train_size)]


    #img_count = 0
    #cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))


    print('saving datasets...')
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((train_datasets, test_datasets), f)
    print('datasets saved at {}'.format(filename))


class SortOfCLEVRDataset(Dataset):
    def __init__(self, root, train=True, transform=None, generate=False, nbrSampledQstPerImg=1):
        super(SortOfCLEVRDataset, self).__init__()
        
        self.root = root
        self.file = 'sort-of-clevr.pickle'
        self.nbrSampledQstPerImg = nbrSampledQstPerImg

        if generate:
            self._generate()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')


        filepath = os.path.join(self.root, self.file)
        with open(filepath, 'rb') as f:
          train_datasets, test_datasets = pickle.load(f)
        
        self.train = train 
        if self.train:
            datasets = train_datasets[:2000]            
        else:
            datasets = test_datasets[:2000]

        dataset = []
        nbrImage = 0
        nbrQst = 0 
        for img, relations, norelations in datasets:
            #img = np.swapaxes(img,0,2)
            img = img.transpose((2,0,1))
            nbrImage += 1

            bina = np.power( 2, np.arange(len(relations[0][0])))
            relations = [ ( int( np.sum(bina*qst)), ans) for qst, ans in zip(relations[0],relations[1])]
            
            for _ in range(self.nbrSampledQstPerImg):
                if len(relations) == 0:    break
                idxsample = random.choice(range(len(relations)))
                qst, ans = relations.pop(idxsample)
                dataset.append((img,qst,ans))
                nbrQst += 1
            
            '''

            for qst,ans in zip(relations[0], relations[1]):
                qst = int(np.amax(qst))
                ans = int(np.amax(ans))
                dataset.append((img,qst,ans))
                nbrQst += 1
            for qst,ans in zip(norelations[0], norelations[1]):
                qst = int(np.amax(qst))
                ans = int(np.amax(ans))
                dataset.append((img,qst,ans))
                nbrQst += 1
            '''

        print("Sort-of-CLEVR Dataset: {} unique images.".format(nbrImage))
        print("Sort-of-CLEVR Dataset: {} questions asked.".format(nbrQst))

        nbr_unique_ans = max([d[2] for d in dataset])+1
        self.dataset = [ (d[0], d[2]+nbr_unique_ans*d[1]) for d in dataset]
        
        self.transform = transform 

    def __len__(self) -> int:
        return len(self.dataset)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,self.file))

    def _generate(self):
        """
        Generate the Sort-of-CLEVR dataset if it doesn't exist already.
        """
        if self._check_exists():
            return
        os.makedirs(self.root, exist_ok=True)
        generate_dataset(self.root)

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

        img, target = self.dataset[idx]
        img = (img*255).astype('uint8').transpose((2,1,0))
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
