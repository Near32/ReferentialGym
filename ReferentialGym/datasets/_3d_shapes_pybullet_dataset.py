from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset 
import os

import numpy as np
import pybullet as pb 
import pybullet_data as pb_d 

import random
from PIL import Image 


import matplotlib.pyplot as plt 
from tqdm import tqdm
import pickle 

# Reproducing: 
# http://alumni.media.mit.edu/~wad/color/numbers.html      
# without white...
colors = [
#Black
(0, 0, 0),
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
#Dk. Gray
(87, 87, 87), 
]

shapes = [
    'cylinder',
    'capsule',
    'sphere',
    'cube',
    'torus',
    'teddy',
    'duck',
]


def generate_datapoint(
    latent_one_hot, 
    latent_values, 
    latent_classes,
    img_size,
    nb_shapes,
    nb_colors,
    nb_samples,
    sampled_positions,
    sampled_orientation,
    physicsClient,
    ):
    '''
    :param latent_one_hot: Numpy Array of shape (nb_objects, latent_one_hot_size)
    :param latent_values: Numpy Array of shape (nb_objects, nb_latent_attr). E.g. contains actual pixel positions.
    :param latent_classes: Numpy Array of shape (nb_objects, nb_latent_attr). E.g. contains bucket positions.
    :param img_size: Integer pixel size of the squared image.
    :param nb_shapes: Integer number of possible shapes.
    :param nb_colors: Integer number of possible colors.
    :param nb_samples: Integer number of possible sampled camera position.
    :param sampled_positions: List of Numpy Array of shape (3,) describing the position of the object for each sample index.
    :param sampled_orientation: List of float describing the Y-axis orientation of the object for each sample index.
    :param physicsClient: Integer identifying the physicsClient used by PyBullet.
    '''
    global colors
    global shapes 
    
    color_id = latent_classes[0]
    obj_color = [float(colors[color_id][0])/255,float(colors[color_id][1])/255,float(colors[color_id][2])/255, 1.0]

    shape_id = latent_classes[1]
    obj_shape = shapes[shape_id]


    obj_position = sampled_positions[latent_classes[2]]; obj_position[2] = 0
    obj_orientation = np.zeros(3);  #obj_orientation[0] = np.pi/2
    obj_orientation[2] = sampled_orientation[latent_classes[2]]

    #print('Position:', obj_position)
    #print('Orientation:', obj_orientation)
    
    cam_eye = np.zeros(3);  cam_eye[2] = 7.0;   cam_eye[1]= 10.0
    cam_target = np.zeros(3)
    cam_up = np.zeros(3);   cam_up[1] = -1.0

    def generate(shapeId, position, orientation, color, physicsClient):
        datapath = pb_d.getDataPath()
        pb.setAdditionalSearchPath(datapath)
        pb.resetSimulation(physicsClient) #pb.RESET_USE_DEFORMABLE_WORLD)
        pb.setGravity(0, 0, -9.81)

        planeId = pb.loadURDF("plane.urdf", [0,0,0])

        if 'torus' in shapeId:
            orientation[0] = np.pi/2
            position[2] += 0.5
            frame_offset_orientation =np.zeros(3);
            frame_offset_position = [0, 0, 0]
            meshScale = [2.0,2.0,2.0]
            torusVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName="torus.obj", 
                rgbaColor=color,
                meshScale=meshScale, 
                visualFramePosition=frame_offset_position,
                visualFrameOrientation=frame_offset_orientation 
            )

            torusCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH,
                fileName="torus.obj", 
                meshScale=meshScale, 
                collisionFramePosition=frame_offset_position,
                collisionFrameOrientation=frame_offset_orientation
            )

            torusId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=torusCollisionId, 
                baseVisualShapeIndex=torusVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'teddy' in shapeId:
            orientation[0] = np.pi/2
            frame_offset_orientation =np.zeros(3);
            frame_offset_position = [-2, -0.5, -0.5]
            meshScale = [4.0,4.0,4.0]
            teddyVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName="teddy2_VHACD_CHs.obj", 
                rgbaColor=color,
                meshScale=meshScale,
                visualFramePosition=frame_offset_position,
                visualFrameOrientation=frame_offset_orientation 
            )

            teddyCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH,
                fileName="teddy2_VHACD_CHs.obj", 
                meshScale=meshScale, 
                collisionFramePosition=frame_offset_position,
                collisionFrameOrientation=frame_offset_orientation
            )

            teddyId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=teddyCollisionId, 
                baseVisualShapeIndex=teddyVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'duck' in shapeId:
            orientation[0] = np.pi/2
            position[2] = -0.25
            meshScale = [2.0,2.0,2.0]
            duckVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName="duck.obj", 
                rgbaColor=color,
                meshScale=meshScale, 
            )

            duckCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_MESH,
                fileName="duck.obj", 
                meshScale=meshScale, 
            )

            duckId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=duckCollisionId, 
                baseVisualShapeIndex=duckVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'cube' in shapeId:
            position[-1] = 1.0
            cubeVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_BOX,
                #fileName="cube.obj", 
                rgbaColor=color,
                halfExtents=[1.0,1.0,1.0], 
            )

            cubeCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_BOX,
                #fileName="cube.obj", 
                halfExtents=[1.0,1.0,1.0], 
            )

            cubeId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=cubeCollisionId, 
                baseVisualShapeIndex=cubeVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'sphere' in shapeId:
            position[-1] = 1.0
            sphereVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_SPHERE,
                #fileName="sphere_smooth.obj", 
                rgbaColor=color,
                radius=1.0, 
            )

            sphereCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_SPHERE,
                #fileName="sphere_smooth.obj", 
                radius=1.0, 
            )

            sphereId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=sphereCollisionId, 
                baseVisualShapeIndex=sphereVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'capsule' in shapeId:
            position[-1] = 1.0
            orientation[0] = np.pi/2
            
            capsuleVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_CAPSULE,
                #fileName="sphere_smooth.obj", 
                rgbaColor=color,
                radius=1.0,
                length=2.0, 
            #   height=1.0, 
            )

            capsuleCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_CAPSULE,
                #fileName="sphere_smooth.obj", 
                radius=1.0, 
                height=2.0,
            #   height=1.0, 
            )

            capsuleId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=capsuleCollisionId, 
                baseVisualShapeIndex=capsuleVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

        elif 'cylinder' in shapeId:
            position[-1] = 1.0
            
            cylinderVisualId = pb.createVisualShape(
                shapeType=pb.GEOM_CYLINDER,
                #fileName="sphere_smooth.obj", 
                rgbaColor=color,
                radius=0.5, 
                length=2.0, 
            )

            cylinderCollisionId = pb.createCollisionShape(
                shapeType=pb.GEOM_CYLINDER,
                #fileName="sphere_smooth.obj", 
                radius=0.5, 
                height=2.0, 
            )

            cylinderId = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=cylinderCollisionId, 
                baseVisualShapeIndex=cylinderVisualId,
                basePosition=position,
                baseOrientation=pb.getQuaternionFromEuler(orientation) 
            )

            
    generate(
        shapeId=obj_shape,
        position=obj_position,
        orientation=obj_orientation,
        color=obj_color,
        physicsClient=physicsClient,
    )

    def render(size=img_size,
               eye=cam_eye, 
               target=cam_target, 
               up=cam_up,
               fov=45,
               aspect=1.0,
               nearVal=0.1,
               farVal=30.1):
        viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

        projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=nearVal,
            farVal=farVal
        )

        w, h, rgba_img, depth_img, seg_img = pb.getCameraImage(
            width=size,
            height=size,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix
        )

        rgb_img = rgba_img[:, :, :-1]

        return rgb_img

    img = render()

    #img = (img/255.).transpose((2,0,1))
    img = (img).astype('uint8').transpose((2,1,0))
    
    return img


def generate_dataset(root,
                     img_size=32,
                     nb_samples=100,
                     nb_shapes=5,
                     nb_colors=5,
                     ):
    global colors
    global shapes 
    dirs = root 
    
    assert nb_shapes <= len(shapes) and nb_colors <= len(colors)

    colors = colors[:nb_colors]
    shapes = shapes[:nb_shapes]
    samples = [i for i in range(nb_samples)]

    sampled_positions = [np.random.uniform(low=-3,high=3, size=(3)) for _ in range(nb_samples)]
    for i in range(len(sampled_positions)): sampled_positions[i][-1] = 0
    sampled_orientation = [np.random.uniform(low=0,high=2*np.pi) for _ in range(nb_samples)]

    latent_one_hot_repr_sizes = {
        "color":nb_colors, #similar to id
        "shape":nb_shapes,
        "sample":nb_samples,
    }

    one_object_latents_ones_hot_size = sum([v for k,v in latent_one_hot_repr_sizes.items()])
    
    print('building dataset...')
    
    possible_shape_values = np.arange(0,nb_shapes)
    possible_color_values = np.arange(0,nb_colors)
    possible_sample_id_values = np.arange(0,nb_samples)

    dummy_latent_values = np.zeros(3).astype(int)
    dummy_latent_class = np.zeros(3).astype(int)
    # (3, )
    dummy_latent_one_hot = np.zeros(one_object_latents_ones_hot_size).astype(int)
    # (one_object_latents_ones_hot_size, )
    
    img_latent_class = []
    img_latent_values = []
    img_latent_one_hot = []
    
    # Setting up the color when sampling later...:
    one_hot_idx_start = 0
    for color_id in possible_color_values:
        obj_latent_class = dummy_latent_class.copy()
        obj_latent_values = dummy_latent_values.copy()
        obj_latent_one_hot = dummy_latent_one_hot.copy()
    
        obj_latent_class[0] = color_id
        obj_latent_values[0] = color_id
        one_hot_idx_start_color = one_hot_idx_start
        obj_latent_one_hot[one_hot_idx_start_color+color_id] = 1
        for shape_id in possible_shape_values:
            obj_latent_class[1] = shape_id
            obj_latent_values[1] = shape_id
            one_hot_idx_start_shape = one_hot_idx_start_color+nb_colors
            obj_latent_one_hot[one_hot_idx_start_shape+shape_id] = 1
            for sample_id in possible_sample_id_values:
                obj_latent_class[2] = sample_id 
                obj_latent_values[2] = sample_id 
                
                one_hot_idx_start_sample = one_hot_idx_start_shape+nb_shapes
                obj_latent_one_hot[one_hot_idx_start_sample+sample_id] = 1
                    
                img_latent_class.append(obj_latent_class.copy())
                img_latent_values.append(obj_latent_values.copy())
                img_latent_one_hot.append(obj_latent_one_hot.copy())
                
                # Reset:
                obj_latent_one_hot[one_hot_idx_start_sample+sample_id] = 0
            
            # Reset:
            obj_latent_one_hot[one_hot_idx_start_shape+shape_id] = 0
        
        # Reset: done at the beginning of the loop...

    dataset = {
        "imgs":{},
        "latents_values":img_latent_values,
        "latents_classes":img_latent_class,
        "latents_one_hot":img_latent_one_hot,
    }

    print('saving datasets...')
    filename = os.path.join(dirs,'3d_shapes_pybullet_dataset.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((dataset, nb_shapes, nb_colors, nb_samples, sampled_positions, sampled_orientation), f)
    print('datasets saved at {}'.format(filename))

    return dataset, nb_shapes, nb_colors, nb_samples, sampled_positions, sampled_orientation

class _3DShapesPyBulletDataset(Dataset) :
    def __init__(self, 
                 root, 
                 img_size,
                 nb_shapes,
                 nb_colors,
                 nb_samples,
                 train=True, 
                 transform=None, 
                 generate=False,
                 split_strategy=None,
                 ):
        super(_3DShapesPyBulletDataset, self).__init__()
        
        self.root = root
        self.file = '3d_shapes_pybullet_dataset.pickle'
        self.img_size = img_size
        self.nb_shapes = nb_shapes
        self.nb_colors = nb_colors
        self.nb_samples = nb_samples

        self.split_strategy = split_strategy        
        
        self.train = train 
        self.generate = generate
        self.transform = transform 
        
        self.physicsClient = None
        if generate or not self._check_exists():
            if not self._check_exists():
                print('Dataset not found. Let us generate it:')

            dataset, nb_shapes, nb_colors, nb_samples, sampled_positions, sampled_orientation = self._generate(
                root=root,
                img_size=img_size,
                nb_shapes=nb_shapes,
                nb_colors=nb_colors,
                nb_samples=self.nb_samples
            )
        else:
            filepath = os.path.join(self.root, self.file)
            with open(filepath, 'rb') as f:
              dataset, nb_shapes, nb_colors, nb_samples, sampled_positions, sampled_orientation = pickle.load(f)
        
        self.sampled_positions = sampled_positions
        self.sampled_orientation = sampled_orientation

        self.latents_values = np.asarray(dataset['latents_values'])
        #(color, shape, sample_id) :
        self.latents_classes = np.asarray(dataset['latents_classes'])
        self.latents_one_hot = np.asarray(dataset['latents_one_hot'])
        self.test_latents_mask = np.zeros_like(self.latents_classes)
        
        self.imgs = dataset['imgs']
        
        self.targets = np.zeros(len(self.latents_classes))
        for idx, latent_cls in enumerate(self.latents_classes):
            color = latent_cls[0]
            shape = latent_cls[1]
            target = color*self.nb_shapes+shape
            self.targets[idx] = target
        
        if self.split_strategy is not None:
            strategy = self.split_strategy.split('-')
            if 'combinatorial' in self.split_strategy:
                self.counter_test_threshold = int(strategy[0][len('combinatorial'):])
                # (default: 2) Specifies the threshold on the number of latent dimensions
                # whose values match a test value. Below this threshold, samples are used in training.
                # A value of 1 implies a basic train/test split that tests generalization to out-of-distribution values.
                # A value of 2 implies a train/test split that tests generalization to out-of-distribution pairs of values...
                # It implies that test value are encountered but never when combined with another test value.
                # It is a way to test for binary compositional generalization from well known stand-alone test values.
                # A value of 3 tests for ternary compositional generalization from well-known:
                # - stand-alone test values, and
                # - binary compositions of test values.
                
                '''
                With regards to designing axises as primitives:
                
                It implies that all the values on this latent axis are treated as test values
                when combined with a test value on any other latent axis.
                
                N.B.: it is not possible to test for out-of-distribution values in that context...
                N.B.1: It is required that the number of primitive latent axis be one less than
                        the counter_test_thershold, at most.

                A number of fillers along this primitive latent axis can then be specified in front
                of the FP pattern...
                Among the effective indices, those with an ordinal lower or equal to the number of
                filler allowed will be part of the training set.
                '''
                self.latent_dims = {}
                # self.strategy[0] : 'combinatorial'
                # 1: Shape
                self.latent_dims['Shape'] = {'size': self.nb_shapes}
                
                self.latent_dims['Shape']['nbr_fillers'] = 0
                self.latent_dims['Shape']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['Shape']['primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[1].split('FP')[0])
                '''
                self.latent_dims['Shape']['image_wise_primitive'] = ('IWP' in strategy[1])
                if self.latent_dims['Shape']['image_wise_primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[1].split('IWP')[0])
                    assert  self.latent_dims['Shape']['nbr_fillers'] < self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'], \
                            "It seems that the test dataset will be empty."
                '''
                self.latent_dims['Shape']['position'] = 1
                # (color, shape, sample)
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[2]:
                    strategy[2] = strategy[2].split('RemainderToUse')
                    self.latent_dims['Shape']['remainder_use'] = int(strategy[2][1])
                    strategy[2] = strategy[2][0]
                else:
                    self.latent_dims['Shape']['remainder_use'] = 0
                
                self.latent_dims['Shape']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['Shape']['untested'] = True
                    self.latent_dims['Shape']['test_set_divider'] = (self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['Shape']['test_set_divider'] = int(strategy[3])

                # 4: Color
                self.latent_dims['Color'] = {'size': self.nb_colors}
                
                self.latent_dims['Color']['nbr_fillers'] = 0
                self.latent_dims['Color']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['Color']['primitive']:
                    self.latent_dims['Color']['nbr_fillers'] = int(strategy[4].split('FP')[0])
                '''
                self.latent_dims['Color']['image_wise_primitive'] = ('IWP' in strategy[4])
                if self.latent_dims['Color']['image_wise_primitive']:
                    self.latent_dims['Color']['nbr_fillers'] = int(strategy[4].split('IWP')[0])
                    assert  self.latent_dims['Color']['nbr_fillers'] < self.latent_dims['Color']['size']//self.latent_dims['Color']['divider'], \
                            "It seems that the test dataset will be empty."
                '''
                self.latent_dims['Color']['position'] = 0
                #(color, shape, X, Y)
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                if 'RemainderToUse' in strategy[5]:
                    strategy[5] = strategy[5].split('RemainderToUse')
                    self.latent_dims['Color']['remainder_use'] = int(strategy[5][1])
                    strategy[5] = strategy[5][0]
                else:
                    self.latent_dims['Color']['remainder_use'] = 0
                self.latent_dims['Color']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['Color']['untested'] = True
                    self.latent_dims['Color']['test_set_divider'] = (self.latent_dims['Color']['size']//self.latent_dims['Color']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['Color']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['Color']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['Color']['test_set_divider'] = int(strategy[6])
                
                # 7: Sample
                self.latent_dims['Sample'] = {'size': self.nb_samples}
                
                self.latent_dims['Sample']['nbr_fillers'] = 0
                self.latent_dims['Sample']['primitive'] = ('FP' in strategy[7])
                if self.latent_dims['Sample']['primitive']:
                    self.latent_dims['Sample']['nbr_fillers'] = int(strategy[7].split('FP')[0])
                '''
                self.latent_dims['Sample']['image_wise_primitive'] = ('IWP' in strategy[7])
                if self.latent_dims['Sample']['image_wise_primitive']:
                    self.latent_dims['Sample']['nbr_fillers'] = int(strategy[7].split('IWP')[0])
                    assert  self.latent_dims['Sample']['nbr_fillers'] < self.latent_dims['Sample']['size']//self.latent_dims['Sample']['divider'], \
                            "It seems that the test dataset will be empty."

                '''
                self.latent_dims['Sample']['position'] = 2
                #(color, shape, sample)
                # 8: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 10  
                if 'RemainderToUse' in strategy[8]:
                    strategy[8] = strategy[8].split('RemainderToUse')
                    self.latent_dims['Sample']['remainder_use'] = int(strategy[8][1])
                    strategy[8] = strategy[8][0]
                else:
                    self.latent_dims['Sample']['remainder_use'] = 0
                self.latent_dims['Sample']['divider'] = int(strategy[8])
                # 9: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 and 10 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                

                if 'N' in strategy[9]:
                    self.latent_dims['Sample']['untested'] = True
                    self.latent_dims['Sample']['test_set_divider'] = (self.latent_dims['Sample']['size']//self.latent_dims['Sample']['divider'])+10
                elif 'E' in strategy[9]:  
                    self.latent_dims['Sample']['test_set_size_sample_from_end'] = int(strategy[9][1:])
                elif 'S' in strategy[9]:  
                    self.latent_dims['Sample']['test_set_size_sample_from_start'] = int(strategy[9][1:])
                else:  
                    self.latent_dims['Sample']['test_set_divider'] = int(strategy[9])
                
                '''
                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] \
                    or self.latent_dims[k]['image_wise_primitive'] \
                    or 'untested' not in self.latent_dims[k]])
                assert nbr_primitives_and_tested==self.counter_test_threshold
                '''
                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] or 'untested' not in self.latent_dims[k]])
                #assert(nbr_primitives_and_tested==self.counter_test_threshold)

            elif 'compositional' in self.split_strategy:
                shuffle_seed = int(self.split_strategy.split('-')[1])
                self.train_nb_possible_colors = int(self.split_strategy.split('_')[-1])
                assert self.train_nb_possible_colors < self.nb_colors
                
                # From shape to colors:
                shapes = {
                    shape_id:np.roll(np.arange(0,self.nb_colors), shift=idx)
                    for idx, shape_id in enumerate(range(self.nb_shapes))
                }
                
                test_nb_possible_colors = self.nb_colors-self.train_nb_possible_colors
                self.training_shape_2_possible_colors = {
                    shape_id:possible_colors[test_nb_possible_colors:]
                    for shape_id, possible_colors in shapes.items()
                }
                self.testing_shape_2_possible_colors = {
                    shape_id:possible_colors[:test_nb_possible_colors]
                    for shape_id, possible_colors in shapes.items()
                }

        else:
            self.divider = 1
            self.offset = 0

        self.indices = []
        if self.split_strategy is None or 'divider' in self.split_strategy:
            for idx in range(len(self.latents_values)):
                if idx % self.divider == self.offset:
                    self.indices.append(idx)

            self.train_ratio = 0.8
            # Shuffled:
            np.random.shuffle(np.asarray(self.indices))
            
            end = int(len(self.indices)*self.train_ratio)
            if self.train:
                self.indices = self.indices[:end]
            else:
                self.indices = self.indices[end:]

            print(f"Split Strategy: {self.split_strategy} --> d {self.divider} / o {self.offset}")
            print(f"Dataset Size: {len(self.indices)} out of {len(self.latents_values)}: {100*len(self.indices)/len(self.latents_values)}%.")
        elif 'combinatorial' in self.split_strategy:
            indices_latents = list(zip(range(self.latents_classes.shape[0]), self.latents_classes))
            for idx, latent_class in indices_latents:
                effective_test_threshold = self.counter_test_threshold
                counter_test = {}
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[dim_dict['position']]
                    quotient = (dim_class+1)//dim_dict['divider']
                    remainder = (dim_class+1)%dim_dict['divider']
                    if remainder!=dim_dict['remainder_use']:
                        skip_it = True
                        break

                    if dim_dict['primitive']:
                        ordinal = quotient
                        if ordinal > dim_dict['nbr_fillers']:
                            effective_test_threshold -= 1

                    if 'test_set_divider' in dim_dict and quotient%dim_dict['test_set_divider']==0:
                        counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_end' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient > max_quotient-dim_dict['test_set_size_sample_from_end']:
                            counter_test[dim_name] = 1
                    elif 'test_set_size_sample_from_start' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        if quotient <= dim_dict['test_set_size_sample_from_start']:
                            counter_test[dim_name] = 1

                    if dim_name in counter_test:
                        self.test_latents_mask[idx, dim_dict['position']] = 1
                        
                if skip_it: continue


                if self.train:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        continue
                    else:
                        self.indices.append(idx)
                else:
                    if len(counter_test) >= effective_test_threshold:#self.counter_test_threshold:
                        self.indices.append(idx)
                    else:
                        continue

            print(f"Split Strategy: {self.split_strategy}")
            print(self.latent_dims)
            print(f"Dataset Size: {len(self.indices)} out of {len(self.latents_values)} : {100*len(self.indices)/len(self.latents_values)}%.")
            
            assert len(self.indices),\
                "No valid data, maybe try a smaller divider..."

        elif 'compositional' in self.split_strategy:
            self.indices = []
            for idx in range(self.latents_classes.shape[0]):
                shape_id = self.latents_classes[idx][1]
                color_id = self.latents_classes[idx][0]

                color_selection = self.training_shape_2_possible_colors
                if not(self.train): color_selection = self.testing_shape_2_possible_colors
                
                if color_id in color_selection[shape_id]:
                    self.indices.append(idx)

            print(f"Dataset Size: {len(self.indices)} out of {len(self.latents_values)}: {100*len(self.indices)/len(self.latents_values)}%.")

        """
        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        self.latents_one_hot = self.latents_one_hot[self.indices]

        self.test_latents_mask = self.test_latents_mask[self.indices]
        self.targets = self.targets[self.indices]
        """

        #self._generate_all()

        print('Dataset loaded : OK.')
    
    def _save_generated_dataset(self):
        if self._check_exists():
            filepath = os.path.join(self.root, self.file)
            with open(filepath, 'rb') as f:
              dataset, _, _, _, _, _ = pickle.load(f)
            
            dataset["imgs"].update(self.imgs)
        else:
            dataset = {
                "imgs":self.imgs,
                "latents_values":self.latents_values,
                "latents_classes":self.latents_classes,
                "latents_one_hot":self.latents_one_hot,
            }

        print('saving datasets...')
        filename = os.path.join(self.root,self.file)
        with  open(filename, 'wb') as f:
            pickle.dump((dataset, self.nb_shapes, self.nb_colors, self.nb_samples, self.sampled_positions, self.sampled_orientation), f)
        print('Datasets saved at {}'.format(filename))

    def _generate_all(self):
        pbar = tqdm(total=len(self.indices))
        for idx in self.indices:
            pbar.update(1)
            if idx in self.imgs:    continue
            self._generate_datapoint(idx=idx)

    def _generate_datapoint(self, idx):
        latents_values = self.latents_values[idx]
        latents_one_hot = self.latents_one_hot[idx]
        latents_classes = self.latents_classes[idx]
        
        if self.physicsClient is None:
            self.physicsClient = pb.connect(pb.DIRECT)

        rgb_img = generate_datapoint(
            latent_one_hot=latents_one_hot, 
            latent_values=latents_values,
            latent_classes=latents_classes,
            img_size=self.img_size,
            nb_shapes=self.nb_shapes,
            nb_colors=self.nb_colors,
            nb_samples=self.nb_samples,
            sampled_positions=self.sampled_positions,
            sampled_orientation=self.sampled_orientation,
            physicsClient=self.physicsClient,
        )

        self.imgs[idx] = rgb_img

        if all([(index in self.imgs) for index in self.indices]):
            self._save_generated_dataset()
            # will only be called once, when the last element has just been generated, 
            # since this whole function will never be called again after all elements
            # are generated...

        
    def __len__(self) -> int:
        return len(self.indices)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,self.file))

    def _generate(self, 
                  root,
                  img_size,
                  nb_shapes,
                  nb_colors,
                  nb_samples):
        """
        Generate the 3DShapesPyBullet dataset if it doesn't exist already.
        """
        if root is None:
            root = self.root
        os.makedirs(root, exist_ok=True)
        return generate_dataset(
            root=root,
            img_size=img_size,
            nb_shapes=nb_shapes,
            nb_colors=nb_colors,
            nb_samples=nb_samples,
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

    def gettestlatentmask(self, idx):
        if idx >= len(self):
            idx = idx%len(self)
        test_latents_mask = self.test_latents_mask[idx]
        return test_latents_mask

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)

        trueidx = self.indices[idx]

        latent_value = torch.from_numpy(self.getlatentvalue(trueidx))
        latent_class = torch.from_numpy(self.getlatentclass(trueidx))
        latent_one_hot = torch.from_numpy(self.getlatentonehot(trueidx))
        test_latents_mask = torch.from_numpy(self.gettestlatentmask(trueidx))

        if trueidx not in self.imgs:    
            self._generate_datapoint(idx=trueidx)

        img = self.imgs[trueidx]
        target = self.getclass(trueidx)
                
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
            "exp_latents_one_hot_encoded":latent_one_hot,
            "exp_test_latents_masks":test_latents_mask,
        }
        
        return sampled_d
