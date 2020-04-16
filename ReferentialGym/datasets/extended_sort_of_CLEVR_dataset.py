from typing import Dict, List, Tuple
import torch 
from torch.utils.data import Dataset 
import os
import pickle 
import numpy as np
import random
import cv2
from PIL import Image 
from tqdm import tqdm


def generate_dataset(root,
                     dataset_size=10000,
                     img_size=75,
                     object_size=5,
                     nb_objects=6,
                     nb_nr_qs=5,
                     nb_r_qs=7,
                    ):
    '''
    Inspired by: https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py
    '''
    
    '''
    question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
    '''
    question_size = nb_objects+2+max(nb_nr_qs, nb_r_qs) 
    ## nb_objects(==nb_colors) for one-hot vector of color, 2 for question type, max(nb_(n)r_qs) for question subtype

    dirs = root 
    
    colors = [
        (0,0,255),##r
        (0,255,0),##g
        (255,0,0),##b
        (0,156,255),##o
        (128,128,128),##k
        (0,255,255)##y
    ]
    assert(nb_objects<=6)
    colors = colors[:nb_objects]
    
    shapes = [
        "circle",
        "rectangle"
    ]
    
    '''
    # 0, as a class, is a lack of object (no color/ no shape):
    latent_one_hot_repr_sizes = {"color":len(colors)+1,
        "shape":len(shapes)+1,
    }

    size_one_hot_vec_per_object = sum([v for k,v in latent_one_hot_repr_sizes.items()])
    nb_attr_per_object = len(latent_one_hot_repr_sizes)
    '''

    pos_X = np.arange(object_size, img_size-object_size+1, 2*object_size)
    pos_Y = np.arange(object_size, img_size-object_size+1, 2*object_size)
    nb_colors = len(colors)
    nb_shapes = len(shapes)
    nX = len(pos_X)
    nY = len(pos_Y)
    latent_one_hot_repr_sizes = {
        "color":nb_colors, #similar to id
        "shape":nb_shapes,
        "pos_X":nX,
        "pos_Y":nY,
    }

    answer_size = 2+2+nb_objects+max(nX,nY)
    """
    Answer : [
        yes,    0
        no,     1
        rectangle,  2
        circle,     3
        *colors/object_id/count,    4~9
        positional_bucket_id/distance,  10~10+max(nX,nY)
    ]
    """
    
    
    one_object_latents_ones_hot_size = sum([v for k,v in latent_one_hot_repr_sizes.items()])
    
    possible_Y_values = pos_Y 
    possible_X_values = pos_X 
    possible_shape_values = np.arange(0,len(shapes))
    possible_color_values = np.arange(0,len(colors))
    possible_object_id_values = np.arange(0,nb_objects)

    dummy_latent_values = np.zeros(4).astype(int)
    dummy_latent_class = np.zeros(4).astype(int)
    # (4, )
    dummy_latent_one_hot = np.zeros(one_object_latents_ones_hot_size).astype(int)

    def generate_obj_latents(obj):
        '''
        :param obj: (color_id, (cx, cy) , shape_str, bx, by)
        '''
        color_id = obj[0]
        shape_id = 0 if obj[2] == 'r' else 1
        xid = obj[-2]
        posx = obj[1][0]
        yid = obj[-1]
        posy = obj[1][1]

        obj_latent_class = dummy_latent_class.copy()
        obj_latent_values = dummy_latent_values.copy()
        obj_latent_one_hot = dummy_latent_one_hot.copy()

        one_hot_idx_start = 0

        # Color:
        obj_latent_class[0] = color_id
        obj_latent_values[0] = color_id
        obj_latent_one_hot[one_hot_idx_start+color_id] = 1

        # Shape:
        obj_latent_class[1] = shape_id
        obj_latent_values[1] = shape_id
        one_hot_idx_start_shape = one_hot_idx_start+nb_colors
        obj_latent_one_hot[one_hot_idx_start_shape+shape_id] = 1

        # X:
        obj_latent_class[2] = xid 
        obj_latent_values[2] = posx 
        one_hot_idx_start_px = one_hot_idx_start_shape+nb_shapes
        obj_latent_one_hot[one_hot_idx_start_px+xid] = 1
        
        # Y:
        obj_latent_class[3] = yid 
        obj_latent_values[3] = posy 
        one_hot_idx_start_py = one_hot_idx_start_px+nX
        obj_latent_one_hot[one_hot_idx_start_py+yid] = 1
        
        return obj_latent_class, obj_latent_values, obj_latent_one_hot

    def find_pos_side_bucket(coord, pos_side):
        return max(0, coord-1) // (2*object_size)

    def generate_center_coord(objects):
        while True:
            pas = True
            center = np.random.randint(0+object_size, img_size - object_size, 2)        
            if len(objects) > 0:
                for obj in objects:
                    name,c,shape = obj[:3]
                    if ((center - c) ** 2).sum() < ((object_size * 2) ** 2):
                        pas = False
            if pas:
                return center

    def generate_datapoint():
        objects = []
        img = np.ones((img_size,img_size,3)) * 255
        for color_id,color in enumerate(colors[:nb_objects]):  
            center = generate_center_coord(objects)
            bx = find_pos_side_bucket(center[0], pos_X)
            by = find_pos_side_bucket(center[1], pos_Y)
            if random.random()<0.5:
                start = (center[0]-object_size, center[1]-object_size)
                end = (center[0]+object_size, center[1]+object_size)
                cv2.rectangle(img, start, end, color, -1)
                objects.append((color_id,center,'r',bx,by))
            else:
                center_ = (center[0], center[1])
                cv2.circle(img, center_, object_size, color, -1)
                objects.append((color_id,center,'c',bx,by))

        # building latents:
        per_obj_latents = [ generate_obj_latents(obj) for obj in objects]
        img_latent_class, img_latent_values, img_latent_one_hot = [*zip(*per_obj_latents)]
        
        img_latent_class = np.concatenate(img_latent_class, axis=0)
        img_latent_values = np.concatenate(img_latent_values, axis=0)
        img_latent_one_hot = np.concatenate(img_latent_one_hot, axis=0)
        
        objects = [ obj_latent for obj_latent in img_latent_values.reshape((-1,4))]
        objects_class = [ obj_latent_class for obj_latent_class in img_latent_class.reshape((-1,4))]

        rel_questions = {st:[] for st in range(nb_r_qs)}
        norel_questions = {st:[] for st in range(nb_nr_qs)}
        rel_answers = {st:[] for st in range(nb_r_qs)}
        norel_answers = {st:[] for st in range(nb_nr_qs)}

        original_question = np.zeros((question_size))

        """Non-relational questions"""
        for subtype_id in range(nb_nr_qs):
            for color_object_id in range(len(colors)):
                question = original_question.copy()
                # What color is the object we are considering, 
                # i.e. which object are we considering? 
                question[color_object_id] = 1
                # non-relational question
                question[nb_objects] = 1
                # subtype :
                question[nb_objects+2+subtype_id] = 1
                """
                Answer : [yes, no, 1~nb_objects(shapes), 1~nb_objects(count)]
                """
                if subtype_id == 0:
                    """query shape->1~nb_shape"""
                    # Account for yes/no :
                    answer_idx = 2+objects[color_object_id][1] 
                    #from idx 0 to nb_shape-1
                elif subtype_id == 1:
                    """query horizontal (X) position->yes/no"""
                    if objects[color_object_id][2] < img_size / 2:
                        answer_idx = 0
                        # yes
                    else:
                        answer_idx = 1
                        # no
                elif subtype_id == 2:
                    """query vertical (Y) position->yes/no"""
                    if objects[color_object_id][3] < img_size / 2:
                        answer_idx = 0
                        # yes
                    else:
                        answer_idx = 1
                        # no
                elif subtype_id == 3:
                    """query horizontal (X) bucket position->yes/no"""
                    answer_idx = 2+nb_shapes+nb_objects+ objects_class[color_object_id][2]
                elif subtype_id == 4:
                    """query vertical (Y) bucket position->yes/no"""
                    answer_idx = 2+nb_shapes+nb_objects+ objects_class[color_object_id][3]
                
                norel_questions[subtype_id].append(question)
                norel_answers[subtype_id].append(answer_idx)
        
        """Relational questions"""
        for subtype_id in range(nb_r_qs):
            for color_object_id in range(len(colors)):
                question = original_question.copy()
                # What color is the object we are considering, 
                # i.e. what object are we considering? 
                question[color_object_id] = 1
                # relational question
                question[nb_objects+1] = 1
                # subtype :
                question[nb_objects+2+subtype_id] = 1
                """
                Answer : [yes, no, 1~nb_shapes, 1~nb_objects(count)]
                """
                if subtype_id == 0:
                    """
                    closest-to->1~nb_shapes
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx, obj in enumerate(objects)]
                    # We make sure that we are not going to sample the object we are considering:
                    dist_list[dist_list.index(0)] = 999
                    closest_id_in_dist_list = dist_list.index(min(dist_list))
                    closest = objects[closest_id_in_dist_list][0]
                    closest_shape_id = objects[closest][1]
                    answer_idx = 2+closest_shape_id
                elif subtype_id == 1:
                    """
                    furthest-from->1~nb_shapes
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx,obj in enumerate(objects)]
                    furthest_id_in_dist_list = dist_list.index(max(dist_list))
                    furthest = objects[furthest_id_in_dist_list][0]
                    furthest_shape_id = objects[furthest][1]
                    answer_idx = 2+furthest_shape_id
                elif subtype_id == 2:
                    """
                    count-same-shape->1~nb_objects(count)
                    """
                    my_obj_shape_id = objects[color_object_id][1]
                    count = -1
                    for obj_id, obj in enumerate(objects):
                        if obj[1] == my_obj_shape_id:
                            count +=1 
                    answer_idx = 2+nb_shapes+count
                    # from idx 2+nb_objects (i.e. count=0, 
                    # which is actually 1 object of the given shape, 
                    # obtained when checking that very object from 
                    # the list of objects ...)
                    # to idx 2+nb_objects + (nb_objects-1) = 3 + nb_objects
                    # (i.e. count=nb_objects-1,
                    # which is actually nb_objects objects of the given shape).
                elif subtype_id == 3:
                    """
                    bucket distance in X to closest-to->1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx, obj in enumerate(objects)]
                    # We make sure that we are not going to sample the object we are considering:
                    dist_list[dist_list.index(0)] = 999
                    closest_id_in_dist_list = dist_list.index(min(dist_list))
                    closest = objects[closest_id_in_dist_list][0]
                    closest_Xbucket = objects_class[closest][2]
                    bucket_distance = np.abs(closest_Xbucket - objects_class[color_object_id][2])
                    answer_idx = 2+nb_shapes+nb_objects+bucket_distance
                elif subtype_id == 4:
                    """
                    bucket distance in Y to closest-to->1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx, obj in enumerate(objects)]
                    # We make sure that we are not going to sample the object we are considering:
                    dist_list[dist_list.index(0)] = 999
                    closest_id_in_dist_list = dist_list.index(min(dist_list))
                    closest = objects[closest_id_in_dist_list][0]
                    closest_Ybucket = objects_class[closest][3]
                    bucket_distance = np.abs(closest_Ybucket - objects_class[color_object_id][3])
                    answer_idx = 2+nb_shapes+nb_objects+bucket_distance
                
                elif subtype_id == 5:
                    """
                    bucket distance in X to furthest-from->1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx,obj in enumerate(objects)]
                    furthest_id_in_dist_list = dist_list.index(max(dist_list))
                    furthest = objects[furthest_id_in_dist_list][0]
                    furthest_Xbucket = objects_class[furthest][2]
                    bucket_distance = np.abs(furthest_Xbucket - objects_class[color_object_id][2])
                    answer_idx = 2+nb_shapes+nb_objects+bucket_distance
                elif subtype_id == 6:
                    """
                    bucket distance in Y to furthest-from->1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                for idx,obj in enumerate(objects)]
                    furthest_id_in_dist_list = dist_list.index(max(dist_list))
                    furthest = objects[furthest_id_in_dist_list][0]
                    furthest_Ybucket = objects_class[furthest][3]
                    bucket_distance = np.abs(furthest_Ybucket - objects_class[color_object_id][3])
                    answer_idx = 2+nb_shapes+nb_objects+bucket_distance
                    
                rel_questions[subtype_id].append(question)
                rel_answers[subtype_id].append(answer_idx)

        # Dict of keys 0,1,2 (subtypes) and values are list of questions (one_hot_vec):
        norelations = (norel_questions, norel_answers)
        relations = (rel_questions, rel_answers)
        
        #img = (img/255.).transpose((2,0,1))
        img = (img).astype('uint8').transpose((2,1,0))

        datapoint = (img, 
            relations, 
            norelations, 
            img_latent_class.reshape(-1), 
            img_latent_values.reshape(-1),
            img_latent_one_hot.reshape(-1))
        
        return datapoint

    print('building dataset...')
    dataset = {
        "imgs":[],
        "latents_values":[],
        "latents_classes":[],
        "latents_one_hot":[],
    }
    '''
    "relational_qs_0":[],
        "relational_qs_1":[],
        "relational_qs_2":[],
        "non_relational_qs_0":[],
        "non_relational_qs_1":[],
        "non_relational_qs_2":[],
        "relational_as_0":[],
        "relational_as_1":[],
        "relational_as_2":[],
        "non_relational_as_0":[],
        "non_relational_as_1":[],
        "non_relational_as_2":[],
    '''
    for subtype_id in range(nb_r_qs):
        dataset[f"relational_qs_{subtype_id}"] = []
        dataset[f"relational_as_{subtype_id}"] = []
    for subtype_id in range(nb_nr_qs):
        dataset[f"non_relational_qs_{subtype_id}"] = []
        dataset[f"non_relational_as_{subtype_id}"] = []

    pbar = tqdm(total=dataset_size)
    for _ in range(dataset_size):
        pbar.update(1)

        datapoint = generate_datapoint()
        #(img, relations, norelations, latent_class, latent_values, latent_one_hot)
        dataset['imgs'].append(datapoint[0])
        dataset['latents_classes'].append(datapoint[-3])
        dataset['latents_values'].append(datapoint[-2])
        dataset['latents_one_hot'].append(datapoint[-1])
        
        for subtype_id in range(nb_r_qs):
            dataset[f"relational_qs_{subtype_id}"].append(np.stack(datapoint[1][0][subtype_id]))
            dataset[f"relational_as_{subtype_id}"].append(np.stack(datapoint[1][1][subtype_id]))

        for subtype_id in range(nb_nr_qs):
            dataset[f"non_relational_qs_{subtype_id}"].append(np.stack(datapoint[2][0][subtype_id]))
            dataset[f"non_relational_as_{subtype_id}"].append(np.stack(datapoint[2][1][subtype_id]))

    print('saving dataset...')
    filename = os.path.join(dirs,'xsort-of-clevr.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((dataset,answer_size), f)
    print('dataset saved at {}'.format(filename))

    return dataset, answer_size


class XSortOfCLEVRDataset(Dataset):
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 generate=False,
                 dataset_size=10000,
                 test_size=2000,
                 img_size=75,
                 object_size=5,
                 nb_objects=6,
                 nb_nr_qs=5,
                 nb_r_qs=7,
                 test_id_analogy=False,
                 test_id_analogy_threshold=3,
                 ):
        super(XSortOfCLEVRDataset, self).__init__()
        
        self.root = root
        self.file = 'xsort-of-clevr.pickle'        
        self.transform = transform 
        self.nb_objects = nb_objects
        self.nb_nr_qs = nb_nr_qs
        self.nb_r_qs = nb_r_qs
        self.test_id_analogy = test_id_analogy
        self.test_id_analogy_threshold = test_id_analogy_threshold

        assert  self.test_id_analogy_threshold < self.nb_objects,\
                "Looks like you are trying to test analogy without enough \
                supporting evidence."

        if not self._check_exists():
            if generate:
                dataset, answer_size = self._generate(root=root,
                                       dataset_size=dataset_size,
                                       img_size=img_size,
                                       object_size=object_size,
                                       nb_objects=nb_objects,
                                       nb_nr_qs=self.nb_nr_qs,
                                       nb_r_qs=self.nb_r_qs)
            else:
                raise RuntimeError('Dataset not found.')
        else:
            filepath = os.path.join(self.root, self.file)
            with open(filepath, 'rb') as f:
              dataset, answer_size = pickle.load(f)
        
        self.answer_size = answer_size

        self.train = train 
        # TODO: handle train tes tsplit:

        self.imgs = np.asarray(dataset['imgs'])
        self.latents_values = np.asarray(dataset['latents_values'])
        #(color, shape, X, Y) :
        self.latents_classes = np.asarray(dataset['latents_classes'])
        self.latents_one_hot = np.asarray(dataset['latents_one_hot'])
        
        self.relational_qs = {idx:np.stack(dataset[f'relational_qs_{idx}']) for idx in range(self.nb_r_qs)}
        self.non_relational_qs = {idx:np.stack(dataset[f'non_relational_qs_{idx}']) for idx in range(self.nb_nr_qs)}
        self.relational_as = {idx:np.stack(dataset[f'relational_as_{idx}']) for idx in range(self.nb_r_qs)}
        self.non_relational_as = {idx:np.stack(dataset[f'non_relational_as_{idx}']) for idx in range(self.nb_nr_qs)}

        sampling_indices = np.random.randint(len(self.imgs), size=test_size)
        if self.train:
            sampling_indices = [idx for idx in range(len(self.imgs)) if idx not in sampling_indices]

        self.imgs = self.imgs[sampling_indices]
        self.latents_values = self.latents_values[sampling_indices]
        self.latents_classes = self.latents_classes[sampling_indices]
        self.latents_one_hot = self.latents_one_hot[sampling_indices]

        self.relational_qs = {k:v[sampling_indices] for k,v in self.relational_qs.items()}
        self.non_relational_qs = {k:v[sampling_indices] for k,v in self.non_relational_qs.items()}
        self.relational_as = {k:v[sampling_indices] for k,v in self.relational_as.items()}
        self.non_relational_as = {k:v[sampling_indices] for k,v in self.non_relational_as.items()}

        self.targets = np.zeros(len(self.latents_classes))
        weights = [np.power(2,idx) for idx in range(self.nb_objects)]

        for idx, latent_cls in enumerate(self.latents_classes):
            img_shapes = [latent_cls[idx_shape] 
                            for idx_shape in range(1,self.nb_objects*4, 4)
                        ]
            img_shapes = [sh*w for sh, w in zip(img_shapes,weights)]
            target = sum(img_shapes)
            self.targets[idx] = target


    def __len__(self) -> int:
        return len(self.imgs)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,self.file))

    def _generate(self, 
                  root,
                  dataset_size,
                  img_size,
                  object_size,
                  nb_objects,
                  nb_nr_qs,
                  nb_r_qs):
        """
        Generate the Sort-of-CLEVR dataset if it doesn't exist already.
        """
        if root is None:
            root = self.root
        os.makedirs(root, exist_ok=True)
        return generate_dataset(root=root,
            dataset_size=dataset_size,
            img_size=img_size,
            object_size=object_size,
            nb_objects=nb_objects,
            nb_nr_qs=nb_nr_qs,
            nb_r_qs=nb_r_qs
            )

    def getclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        target = self.targets[idx]
        return target

    def getlatentvalue(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_value = self.latents_values[idx]
        return latent_value

    def getlatentclass(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_class = self.latents_classes[idx]
        return latent_class

    def getlatentonehot(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        latent_one_hot = self.latents_one_hot[idx]
        return latent_one_hot

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)

        img = self.imgs[idx]
        target = self.getclass(idx)
        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_class = torch.from_numpy(self.getlatentclass(idx))
        latent_one_hot = torch.from_numpy(self.getlatentonehot(idx))
                
        relational_questions = {f"relational_questions_{k}":torch.from_numpy(v[idx]).float() for k,v in self.relational_qs.items()}
        non_relational_questions = {f"non_relational_questions_{k}":torch.from_numpy(v[idx]).float() for k,v in self.non_relational_qs.items()}
        #(nbr_objects x qst_repr=15)
        
        relational_answers = {f"relational_answers_{k}":torch.from_numpy(v[idx]).long() for k,v in self.relational_as.items()}
        non_relational_answers = {f"non_relational_answers_{k}":torch.from_numpy(v[idx]).long() for k,v in self.non_relational_as.items()}
        #(nbr_objects x ans_repr=1)

        # Do we test the analogy on the color/object_id?
        if self.test_id_analogy:
            # Let us reserve the QAs with regard to color/object_id greater than the given threshold:
            for (strq,poqs), (stra,poas) in zip(relational_questions.items(), relational_answers.items()):
                if self.train:
                    # Only take the first ones:
                    relational_questions[strq] = poqs[:self.test_id_analogy_threshold,...]
                    relational_answers[stra] = poas[:self.test_id_analogy_threshold,...]
                else:
                    # Only take the last ones:
                    relational_questions[strq] = poqs[self.test_id_analogy_threshold:,...]
                    relational_answers[stra] = poas[self.test_id_analogy_threshold:,...]

            for (strq,poqs), (stra,poas) in zip(non_relational_questions.items(), non_relational_answers.items()):
                if self.train:
                    # Only take the first ones:
                    non_relational_questions[strq] = poqs[:self.test_id_analogy_threshold,...]
                    non_relational_answers[stra] = poas[:self.test_id_analogy_threshold,...]
                else:
                    # Only take the last ones:
                    non_relational_questions[strq] = poqs[self.test_id_analogy_threshold:,...]
                    non_relational_answers[stra] = poas[self.test_id_analogy_threshold:,...]

        #img = (img*255).astype('uint8').transpose((2,1,0))
        img = img.transpose((2,1,0))
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        sampled_d = {
            "experiences":img, 
            "exp_labels":target, 
            "exp_latents":latent_class, 
            "exp_latents_values":latent_value,
            "exp_latents_one_hot":latent_one_hot
        }
        
        sampled_d.update(relational_questions)
        sampled_d.update(non_relational_questions)

        sampled_d.update(relational_answers)
        sampled_d.update(non_relational_answers)

        return sampled_d