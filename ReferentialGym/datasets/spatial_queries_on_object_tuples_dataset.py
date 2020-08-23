from typing import Dict, List, Tuple
import torch 
from torch.utils.data import Dataset 
import os
import pickle 
import copy
import numpy as np
import random
import cv2
from PIL import Image 

import matplotlib.pyplot as plt 
from tqdm import tqdm

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
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]



def generate_datapoint(
    latent_one_hot, 
    latent_values, 
    latent_classes,
    nb_r_qs,
    nb_nr_qs,
    nb_brq_qs,
    img_size,
    nb_objects,
    nb_shapes,
    fontScale,
    thickness,
    font):
    '''
    :param latent_one_hot: Numpy Array of shape (nb_objects, latent_one_hot_size)
    :param latent_values: Numpy Array of shape (nb_objects, nb_latent_attr). E.g. contains actual pixel positions.
    :param latent_classes: Numpy Array of shape (nb_objects, nb_latent_attr). E.g. contains bucket positions.
    :param nb_r_qs: Integer number of relational question subtypes.
    :param nb_nr_qs: Integer number of non-relational question subtypes.
    :param nb_brq_qs: Integer number of binary relational question subtypes.
    :param img_size: Integer pixel size of the squared image.
    :param nb_objects: Integer number of objects/colour in the image.
    :param nb_shapes: Integer number of possible shapes for each object.
    :param fontScale: Float (scale) size of the font used.
    :param thickness: Integer thickness of the characters/shapes when drawn on the image.
    :param font: name of the OpenCV font to use.
    '''
    global colors
    global shapes 

    textSize, _ = cv2.getTextSize("X", font, fontScale, thickness)
    textSizeX, textSizeY = textSize
    #textSizeWidth, textSizeHeight = textSize
    
    object_size = max(textSizeX,textSizeY)
    
    nb_question_types = 3
    question_size = nb_objects+nb_shapes+nb_question_types+max(nb_nr_qs,nb_r_qs,nb_brq_qs)
    
    pos_X = np.arange(object_size, img_size-object_size//2, object_size)
    pos_Y = np.arange(object_size, img_size-object_size//2, object_size)
    
    nb_colors = nb_objects
    nX = len(pos_X)
    nY = len(pos_Y)
    latent_one_hot_repr_sizes = {
        "color":nb_colors, #similar to id
        "shape":nb_shapes,
        "pos_X":nX,
        "pos_Y":nY,
    }

    """
    Answer : [
        yes:
            0
        no:
            1
        shapes:  
            2~nb_shapes+2
        *colors/object_id/count:    
            (nb_shapes+2)+1 ~ (nb_shapes+2)+1+nb_objects
        positional_bucket_id/distance:  
            (nb_shapes+2)+1+nb_objects)+1 ~ (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)
        overlap_situation:  
            (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)+1
        irrelevant_question:
            (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)+2
    ]
    """
    answer2idx = {
        "yes":0,
        "no":1,
        "shape":np.arange(2,nb_shapes+2),
        "count":np.arange((nb_shapes+2), 
            (nb_shapes+2)+nb_objects),
        "distance":np.arange(((nb_shapes+2)+nb_objects), 
            ((nb_shapes+2)+nb_objects)+max(nX,nY)),
        "overlap_situation":((nb_shapes+2)+nb_objects)+max(nX,nY),
        "irrelevant_question":((nb_shapes+2)+nb_objects)+max(nX,nY)+1,
    }
    nb_answers = answer2idx["irrelevant_question"]+1


    objects = []
    # [color, shape, xpos, ypos]
    img = np.ones((img_size,img_size,3),dtype='uint8') * 255
    for color_object_id, object_values in enumerate(latent_values):  
        objects.append(object_values)

        assert(color_object_id==object_values[0])

        color = colors[object_values[0]]
        shape = shapes[object_values[1]]

        # Draw a white square first, 
        # to make the drawing clear enough,
        # in case of overlapping:
        start = (object_values[2]-object_size//2, object_values[3]-object_size//2)
        end = (object_values[2]+object_size//2, object_values[3]+object_size//2)
        img = cv2.rectangle(img, start, end, (255,255,255), -1)
        
        lowerLeftCorner = object_values[2:]
        lowerLeftCorner = (lowerLeftCorner[0]-object_size//2, lowerLeftCorner[1]+object_size//2)   
        img = cv2.putText(img, shape, lowerLeftCorner, font, fontScale, color, thickness, cv2.LINE_AA)

    rel_questions = {st:[] for st in range(nb_r_qs)}
    norel_questions = {st:[] for st in range(nb_nr_qs)}
    
    rel_answers = {st:[] for st in range(nb_r_qs)}
    norel_answers = {st:[] for st in range(nb_nr_qs)}

    birelq_questions = {st:[] for st in range(nb_brq_qs)}
    birelq_answers = {st:[] for st in range(nb_brq_qs)}
    
    original_question = np.zeros((question_size))

    # overlapping?
    object_overlapped = [False for _ in range(nb_objects)]
    for idx1, obj1 in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if idx2 <= idx1: continue
            # Drawing happens in the order of the objects
            # So only the latest objects can be drawn over some
            # other object...
            if obj2[2] == obj1[2] and obj2[3] == obj1[3]:
                    object_overlapped[idx1] = True
                    break
                    # one overlap is sufficient 
                    # to prevent from distinguishing this object...
    
    """Non-relational questions"""
    for subtype_id in range(nb_nr_qs):
        for color_object_id in range(nb_objects):
            question = original_question.copy()
            # What color is the object we are considering, 
            # i.e. which object are we considering? 
            question[color_object_id] = 1
            # shape argument is left with zeros everywhere since not relevant
            # non-relational question
            question[nb_objects+nb_shapes] = 1
            # subtype :
            question[nb_objects+nb_shapes+nb_question_types+subtype_id] = 1
            if object_overlapped[color_object_id]:
                # We cannot discernate it, so we output "overlap" answer:
                answer_idx = answer2idx["overlap_situation"]
            else:
                if subtype_id == 0:
                    """query shape->1~nb_shape"""
                    answer_idx = answer2idx["shape"][objects[color_object_id][1]] 
                    #from idx 0 to nb_shape-1
                elif subtype_id == 1:
                    """query horizontal (X) position->yes/no"""
                    if objects[color_object_id][2] < img_size / 2:
                        answer_idx = answer2idx["yes"]
                    else:
                        answer_idx = answer2idx["no"]
                elif subtype_id == 2:
                    """query vertical (Y) position->yes/no"""
                    if objects[color_object_id][3] < img_size / 2:
                        answer_idx = answer2idx["yes"]
                    else:
                        answer_idx = answer2idx["no"]
                elif subtype_id == 3:
                    """query horizontal (X) bucket position->position/distance answer"""
                    answer_idx = answer2idx["distance"][latent_classes[color_object_id][2]]
                elif subtype_id == 4:
                    """query vertical (Y) bucket position->position/distance answer"""
                    answer_idx = answer2idx["distance"][latent_classes[color_object_id][3]]                    
                
            norel_questions[subtype_id].append(question)
            norel_answers[subtype_id].append(answer_idx)
    
    """Relational questions"""
    non_overlapped_objects = [(idx,obj) for idx,obj in enumerate(objects) if not(object_overlapped[idx])]
    for subtype_id in range(nb_r_qs):
        for color_object_id in range(nb_objects):
            question = original_question.copy()
            # What color is the object we are considering, 
            # i.e. what object are we considering? 
            question[color_object_id] = 1
            # shape argument is left with zeros everywhere since not relevant
            # relational question
            question[nb_objects+nb_shapes+1] = 1
            # subtype :
            question[nb_objects+nb_shapes+nb_question_types+subtype_id] = 1
            if object_overlapped[color_object_id]:
                # We cannot discernate the argument object, so we output "overlap" answer:
                answer_idx = answer2idx["overlap_situation"]
            else:
                if subtype_id == 0:
                    """
                    shape-of-closest-to (among non-overlapped objects) -> 1~nb_shapes
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx, obj in non_overlapped_objects]
                        # We make sure that we are not going to sample the object we are considering:
                        dist_list[dist_list.index(0)] = 999
                        closest_id_in_dist_list = dist_list.index(min(dist_list))
                        closest = non_overlapped_objects[closest_id_in_dist_list][0]
                        closest_shape_id = objects[closest][1]
                        answer_idx = answer2idx["shape"][closest_shape_id]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]
                elif subtype_id == 1:
                    """
                    shape-of-furthest-from (among non-overlapped objects) -> 1~nb_shapes
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx,obj in non_overlapped_objects]
                        furthest_id_in_dist_list = dist_list.index(max(dist_list))
                        furthest = non_overlapped_objects[furthest_id_in_dist_list][0]
                        furthest_shape_id = objects[furthest][1]
                        answer_idx = answer2idx["shape"][furthest_shape_id]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]
                elif subtype_id == 2:
                    """
                    count-same-shape (among non-overlapped objects) -> 1~nb_objects(count)
                    """
                    my_obj_shape_id = objects[color_object_id][1]
                    count = 0
                    for obj_id, obj in non_overlapped_objects:
                        if obj[1] == my_obj_shape_id:
                            count +=1 
                    answer_idx = answer2idx["count"][count-1]
                    # from count to idx of the answer
                elif subtype_id == 3:
                    """
                    distance in X of closest-to (among non-overlapped objects) -> 1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx, obj in non_overlapped_objects]
                        # We make sure that we are not going to sample the object we are considering:
                        dist_list[dist_list.index(0)] = 999
                        closest_id_in_dist_list = dist_list.index(min(dist_list))
                        closest = non_overlapped_objects[closest_id_in_dist_list][0]
                        closest_Xbucket = latent_classes[closest][2]
                        my_obj_Xbucket = latent_classes[color_object_id][2]
                        bucket_distance = np.abs(closest_Xbucket - my_obj_Xbucket)
                        answer_idx = answer2idx["distance"][bucket_distance]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]
                elif subtype_id == 4:
                    """
                    distance in Y of closest-to (among non-overlapped objects) -> 1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx, obj in non_overlapped_objects]
                        # We make sure that we are not going to sample the object we are considering:
                        dist_list[dist_list.index(0)] = 999
                        closest_id_in_dist_list = dist_list.index(min(dist_list))
                        closest = non_overlapped_objects[closest_id_in_dist_list][0]
                        closest_Ybucket = latent_classes[closest][3]
                        my_obj_Ybucket = latent_classes[color_object_id][3]
                        bucket_distance = np.abs(closest_Ybucket - my_obj_Ybucket)
                        answer_idx = answer2idx["distance"][bucket_distance]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]

                elif subtype_id == 5:
                    """
                    distance in X of furthest-to (among non-overlapped objects) -> 1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx, obj in non_overlapped_objects]
                        furthest_id_in_dist_list = dist_list.index(max(dist_list))
                        furthest = non_overlapped_objects[furthest_id_in_dist_list][0]
                        furthest_Xbucket = latent_classes[furthest][2]
                        my_obj_Xbucket = latent_classes[color_object_id][2]
                        bucket_distance = np.abs(furthest_Xbucket - my_obj_Xbucket)
                        answer_idx = answer2idx["distance"][bucket_distance]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]
                elif subtype_id == 6:
                    """
                    distance in Y of furthest-to (among non-overlapped objects) -> 1~max(nX,nY) (distance)
                    """
                    my_obj_pos = np.asarray([objects[color_object_id][2],objects[color_object_id][3]])
                    if len(non_overlapped_objects)>1:
                        dist_list = [((my_obj_pos - np.asarray([obj[2],obj[3]])) ** 2).sum() 
                                    for idx, obj in non_overlapped_objects]
                        furthest_id_in_dist_list = dist_list.index(max(dist_list))
                        furthest = non_overlapped_objects[furthest_id_in_dist_list][0]
                        furthest_Ybucket = latent_classes[furthest][3]
                        my_obj_Ybucket = latent_classes[color_object_id][3]
                        bucket_distance = np.abs(furthest_Ybucket - my_obj_Ybucket)
                        answer_idx = answer2idx["distance"][bucket_distance]
                    else:
                        # We cannot discernate any other objects, so we output "overlap" answer:
                        answer_idx = answer2idx["overlap_situation"]

            rel_questions[subtype_id].append(question)
            rel_answers[subtype_id].append(answer_idx)


    """Binary Relational Query questions"""
    shape2nonoverlapped_obj_ids = {
        shape_id:[obj_id for obj_id in range(nb_objects) if not(object_overlapped[obj_id])]
        for shape_id in range(nb_shapes)
    }
    
    for subtype_id in range(nb_brq_qs):
        for color_object_id in range(nb_objects):
            shape_pred_qs = []
            shape_pred_as = []
            for shape_argument_id in range(nb_shapes):
                question = original_question.copy()
                # What color is the object we are considering, 
                # i.e. what object are we considering? 
                question[color_object_id] = 1
                # WARNING: this time, the shape argument is necessary:
                question[nb_objects+shape_argument_id] = 1
                # binary relational query question
                question[nb_objects+nb_shapes+2] = 1
                # subtype :
                question[nb_objects+nb_shapes+nb_question_types+subtype_id] = 1
                if object_overlapped[color_object_id]:
                    # We cannot discernate the argument object, so we output "overlap" answer:
                    answer_idx = answer2idx["overlap_situation"]
                # Is it an irrelevant question?
                # I.e. none of the visible objects display the argument shape:
                elif len([obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] if obj_id != color_object_id])==0:
                    answer_idx = answer2idx["irrelevant_question"]
                else:
                    # We are certain that the question is relevant.
                    # I.e. at least one visible object 
                    # --that is not the focus of the question--
                    # is of the argument shape.
                    
                    # ANY?

                    if subtype_id == 0:
                        """
                        “Is there ANY object with shape [predicate:shape_id] on the LEFT OF object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Xpos = objects[color_object_id][2]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Xpos_withArgShape = [objects[obj_id][2] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_left_of_my_obj = [ooX <= my_obj_Xpos for ooX in other_obj_Xpos_withArgShape]
                        if any(is_other_obj_left_of_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    elif subtype_id == 1:
                        """
                        “Is there ANY object with shape [predicate:shape_id] on the RIGHT OF object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Xpos = objects[color_object_id][2]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Xpos_withArgShape = [objects[obj_id][2] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_right_of_my_obj = [ooX >= my_obj_Xpos for ooX in other_obj_Xpos_withArgShape]
                        if any(is_other_obj_right_of_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]

                    elif subtype_id == 2:
                        """
                        “Is there ANY object with shape [predicate:shape_id] ABOVE object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Ypos = objects[color_object_id][3]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Ypos_withArgShape = [objects[obj_id][3] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_above_my_obj = [ooY <= my_obj_Ypos for ooY in other_obj_Ypos_withArgShape]
                        if any(is_other_obj_above_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    elif subtype_id == 3:
                        """
                        “Is there ANY object with shape [predicate:shape_id] BELOW object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Ypos = objects[color_object_id][3]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Ypos_withArgShape = [objects[obj_id][3] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_below_my_obj = [ooY >= my_obj_Ypos for ooY in other_obj_Ypos_withArgShape]
                        if any(is_other_obj_below_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    
                    # ALL?

                    elif subtype_id == 4:
                        """
                        “Are ALL objects with shape [predicate:shape_id] on the LEFT OF object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Xpos = objects[color_object_id][2]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Xpos_withArgShape = [objects[obj_id][2] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_left_of_my_obj = [ooX <= my_obj_Xpos for ooX in other_obj_Xpos_withArgShape]
                        if all(is_other_obj_left_of_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    elif subtype_id == 5:
                        """
                        “Are ALL object with shape [predicate:shape_id] on the RIGHT OF object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Xpos = objects[color_object_id][2]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Xpos_withArgShape = [objects[obj_id][2] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_right_of_my_obj = [ooX >= my_obj_Xpos for ooX in other_obj_Xpos_withArgShape]
                        if all(is_other_obj_right_of_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]

                    elif subtype_id == 6:
                        """
                        “Are ALL object with shape [predicate:shape_id] ABOVE object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Ypos = objects[color_object_id][3]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Ypos_withArgShape = [objects[obj_id][3] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_above_my_obj = [ooY <= my_obj_Ypos for ooY in other_obj_Ypos_withArgShape]
                        if all(is_other_obj_above_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    elif subtype_id == 7:
                        """
                        “Are ALL object with shape [predicate:shape_id] BELOW object
                        [predicate argument:object_id]? 
                        [subject:truth_value_id]”
                        """
                        my_obj_Ypos = objects[color_object_id][3]
                        other_obj_ids_withArgShape = [
                            obj_id for obj_id in shape2nonoverlapped_obj_ids[shape_argument_id] 
                                if obj_id != color_object_id
                                and objects[obj_id][1] == shape_argument_id
                        ]
                        other_obj_Ypos_withArgShape = [objects[obj_id][3] for obj_id in other_obj_ids_withArgShape]
                        is_other_obj_below_my_obj = [ooY >= my_obj_Ypos for ooY in other_obj_Ypos_withArgShape]
                        if all(is_other_obj_below_my_obj):
                            answer_idx = answer2idx["yes"]
                        else:
                            answer_idx = answer2idx["no"]
                    
                shape_pred_qs.append(question)
                shape_pred_as.append(np.asarray([answer_idx]))

            birelq_questions[subtype_id].append(np.stack(shape_pred_qs))
            # nb_shapes x question_size
            birelq_answers[subtype_id].append(np.stack(shape_pred_as))
            # nb_shapes x 1


    # Dict of keys (subtypes) and values are list of questions (one_hot_vec):
    norelations = (norel_questions, norel_answers)
    relations = (rel_questions, rel_answers)
    
    birelations = (birelq_questions, birelq_answers)
    
    #img = (img/255.).transpose((2,0,1))
    img = (img).astype('uint8').transpose((2,1,0))
    
    datapoint = (img, 
        relations, 
        norelations, 
        birelations,)
    
    return datapoint

def generate_dataset(root,
                     nb_nr_qs=5,
                     nb_r_qs=7,
                     nb_brq_qs=8,
                     img_size=32,
                     nb_objects=2,
                     nb_shapes=6,
                     font=cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=0.5,
                     thickness=1,
                     random_generation=False,
                     nb_samples=None,
                     care_about_overlap=True,
                    ):
    global colors
    global shapes 

    textSize, _ = cv2.getTextSize("X", font, fontScale, thickness)
    textSizeX, textSizeY = textSize
    #textSizeWidth, textSizeHeight = textSize
    
    object_size = max(textSizeX,textSizeY)
    
    nb_question_types = 3
    question_size = nb_objects+nb_shapes+3+max(nb_nr_qs,nb_r_qs,nb_brq_qs)

    '''
    nb_objects(==nb_colors) for one-hot vector of color: 
    --> object identifier in the question.
    nb_shapes for one-hot vector of shape:
    --> shape identifier in the binary relational query questions, it is an extra predicate argument.
    +3 for question type: binary relational queries(2) or relational(1) or non-relational(0)
    + nb_question_subtypes...
    '''
    dirs = root 
    
    assert(nb_objects <= len(colors))
    colors = colors[:nb_objects]
    
    shapes = shapes[:nb_shapes]

    pos_X = np.arange(object_size, img_size-object_size//2, object_size)
    pos_Y = np.arange(object_size, img_size-object_size//2, object_size)
    
    nb_colors = len(colors)
    nb_shapes = len(shapes)
    nX = len(pos_X)
    nY = len(pos_Y)
    
    print(nX, nY)

    latent_one_hot_repr_sizes = {
        "color":nb_colors, #similar to id
        "shape":nb_shapes,
        "pos_X":nX,
        "pos_Y":nY,
    }

    """
    Answer : [
        yes:
            0
        no:
            1
        shapes:  
            2~nb_shapes+2
        *colors/object_id/count:    
            (nb_shapes+2)+1 ~ (nb_shapes+2)+1+nb_objects
        positional_bucket_id/distance:  
            (nb_shapes+2)+1+nb_objects)+1 ~ (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)
        overlap_situation:  
            (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)+1
        irrelevant_question:
            (nb_shapes+2)+1+nb_objects)+1+max(nX,nY)+2
    ]
    """
    answer2idx = {
        "yes":0,
        "no":1,
        "shape":np.arange(2,nb_shapes+2),
        "count":np.arange(
            (nb_shapes+2), 
            (nb_shapes+2)+nb_objects
        ),
        "distance":np.arange(
            ((nb_shapes+2)+nb_objects), 
            ((nb_shapes+2)+nb_objects)+max(nX,nY)
        ),
        "overlap_situation":((nb_shapes+2)+nb_objects)+max(nX,nY),
        "irrelevant_question":((nb_shapes+2)+nb_objects)+max(nX,nY)+1,
    }
    
    idx2answer = {}
    for answer_key, values in answer2idx.items():
        if isinstance(values, int):
            idx2answer[values] = answer_key
        else:
            init_value = None
            for value in values:
                if init_value is None: init_value = value
                idx2answer[value] = answer_key+str(value-init_value+1)
        

    nb_answers = answer2idx["irrelevant_question"]+1

    one_object_latents_ones_hot_size = sum([v for k,v in latent_one_hot_repr_sizes.items()])
    
    print('building dataset...')
    
    possible_Y_values = pos_Y 
    possible_X_values = pos_X 
    possible_shape_values = np.arange(0,len(shapes))
    possible_color_values = np.arange(0,len(colors))
    possible_object_id_values = np.arange(0,nb_objects)

    dummy_latent_values = np.zeros(4).astype(int)
    dummy_latent_class = np.zeros(4).astype(int)
    # (4, )
    dummy_latent_one_hot = np.zeros(one_object_latents_ones_hot_size).astype(int)
    # (one_object_latents_ones_hot_size, )
    
    latent_class_per_obj_list = []
    latent_values_per_obj_list = []
    latent_one_hot_per_obj_list = []
    
    # Setting up the color when sampling later...:
    one_hot_idx_start = 0
    for shape_id in possible_shape_values:

        obj_latent_class = dummy_latent_class.copy()
        obj_latent_values = dummy_latent_values.copy()
        obj_latent_one_hot = dummy_latent_one_hot.copy()
    
        obj_latent_class[1] = shape_id
        obj_latent_values[1] = shape_id
        one_hot_idx_start_shape = one_hot_idx_start+nb_colors
        obj_latent_one_hot[one_hot_idx_start_shape+shape_id] = 1
        for xid, posx in enumerate(possible_X_values):
            obj_latent_class[2] = xid 
            obj_latent_values[2] = posx 
            one_hot_idx_start_px = one_hot_idx_start_shape+nb_shapes
            obj_latent_one_hot[one_hot_idx_start_px+xid] = 1
            for yid, posy in enumerate(possible_Y_values):
                obj_latent_class[3] = yid 
                obj_latent_values[3] = posy 
                
                one_hot_idx_start_py = one_hot_idx_start_px+nX
                obj_latent_one_hot[one_hot_idx_start_py+yid] = 1
                    
                latent_class_per_obj_list.append(obj_latent_class.copy())
                latent_values_per_obj_list.append(obj_latent_values.copy())
                latent_one_hot_per_obj_list.append(obj_latent_one_hot.copy())
                
                # Reset:
                obj_latent_one_hot[one_hot_idx_start_py+yid] = 0
            
            # Reset:
            obj_latent_one_hot[one_hot_idx_start_px+xid] = 0
        
        # Reset: done at the beginning of loop...

    nbr_images = np.power((nb_shapes*nX*nY), nb_objects)
    dataset = {
        "imgs":[],
        "latents_values":[],
        "latents_classes":[],
        "latents_one_hot":[],
    }

    for subtype_id in range(nb_r_qs):
        dataset[f"relational_qs_{subtype_id}"] = []
        dataset[f"relational_as_{subtype_id}"] = []
    
    for subtype_id in range(nb_nr_qs):
        dataset[f"non_relational_qs_{subtype_id}"] = []
        dataset[f"non_relational_as_{subtype_id}"] = []

    for subtype_id in range(nb_brq_qs):
        dataset[f"binary_relational_query_qs_{subtype_id}"] = []
        dataset[f"binary_relational_query_as_{subtype_id}"] = []

    if random_generation:
        object_id_to_idx = {oid:np.random.randint(len(latent_class_per_obj_list)) for oid in range(nb_objects)}
        
        # Balance the shapes: sample from weighted distribution 
        # whose weights are inversely proportional to the current number of instances.
        # TODO:
        shape_nb_instances = np.zeros(nb_shapes)
        mean_nb_instances_per_shape = nb_samples*nb_objects // nb_shapes
        shape_distr = mean_nb_instances_per_shape - shape_nb_instances

        ptr_object_id = nb_objects-1
        continuer = True

        pbar = tqdm(total=nb_samples)
        for sample_idx in range(nb_samples):
            pbar.update(1)

            novel_sample = False 
            overlap = False
            while not novel_sample or overlap:
                object_id_to_idx[ptr_object_id] = np.random.randint(len(latent_class_per_obj_list))
                    
                object_id_to_latent_values = {oid:latent_values_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}
                object_id_to_latent_classes = {oid:latent_class_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}
                object_id_to_latent_one_hot = {oid:latent_one_hot_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}

                positions = np.stack([lc[2:] for lc in object_id_to_latent_classes.values()])
                for idx1, p1 in enumerate(positions[:-1]):
                    for idx2, p2 in enumerate(positions[idx1+1:]):
                        if all(p1 == p2):
                            overlap = True 
                            # let us focus on one of the issue: in case we initialise in a wrong configuration...
                            ptr_object_id = idx1
                            break
                overlap = care_about_overlap and overlap
                if overlap: 
                    overlap = False
                    continue 

                # Setting up the color:
                for oid in object_id_to_latent_values:
                    object_id_to_latent_classes[oid][0] = oid
                    object_id_to_latent_values[oid][0] = oid
                    object_id_to_latent_one_hot[oid][oid] = 1

                latent_one_hot = np.stack(list(object_id_to_latent_one_hot.values()))
                #latent_values = np.stack(list(object_id_to_latent_values.values()))
                #latent_classes = np.stack(list(object_id_to_latent_classes.values()))

                novel_sample = latent_one_hot not in dataset['latents_one_hot']
                
                
            img_latent_classes = np.stack(list(object_id_to_latent_classes.values()))
            # (nb_objects, 4)
            img_latent_values = np.stack(list(object_id_to_latent_values.values()))
            # (nb_objects, 4)
            img_latent_one_hot = np.stack(list(object_id_to_latent_one_hot.values()))
            # (nb_objects, one_object_latents_ones_hot_size)
            
            dataset['latents_values'].append(img_latent_values.reshape(-1))
            dataset['latents_one_hot'].append(img_latent_one_hot.reshape(-1))
            dataset['latents_classes'].append(img_latent_classes.reshape(-1))
            

            # At each image generation, the ptr_object_id is moved circularly
            # The object pointed by the pointer ptr_object_id has its attributes randomized.
            # This random generation approach furthers the existance of contrastive samples.
            # For each sample, there exist at least one sample whose configuration only differs
            # with regards to the attributes of one object.

            ptr_object_id -= 1
            if ptr_object_id<0:
                ptr_object_id = nb_objects-1

    else:
        object_id_to_idx = {oid:0 for oid in range(nb_objects)}
        continuer = True

        pbar = tqdm(total=nbr_images)
        while continuer:
            pbar.update(1)

            object_id_to_latent_values = {oid:latent_values_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}
            object_id_to_latent_classes = {oid:latent_class_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}
            object_id_to_latent_one_hot = {oid:latent_one_hot_per_obj_list[oidx].copy() for oid,oidx in object_id_to_idx.items()}
                    
            # Setting up the color:
            for oid in object_id_to_latent_values:
                object_id_to_latent_classes[oid][0] = oid
                object_id_to_latent_values[oid][0] = oid
                object_id_to_latent_one_hot[oid][oid] = 1

            img_latent_classes = np.stack(list(object_id_to_latent_classes.values()))
            # (nb_objects, 4)
            img_latent_values = np.stack(list(object_id_to_latent_values.values()))
            # (nb_objects, 4)
            img_latent_one_hot = np.stack(list(object_id_to_latent_one_hot.values()))
            # (nb_objects, one_object_latents_ones_hot_size)
            
            dataset['latents_values'].append(img_latent_values.reshape(-1))
            dataset['latents_one_hot'].append(img_latent_one_hot.reshape(-1))
            dataset['latents_classes'].append(img_latent_classes.reshape(-1))
            

            object_id_to_idx[nb_objects-1] = (object_id_to_idx[nb_objects-1]+1)%len(latent_class_per_obj_list)
            ptr_object_id = nb_objects-1
            while object_id_to_idx[ptr_object_id]==0:
                ptr_object_id -= 1

                '''
                If we want to look at id -1,
                then it means that id 0 (and others) has 
                seen all the possible object values,
                so we are done:
                '''
                if ptr_object_id < 0:
                    continuer = False
                    break
                else:
                    object_id_to_idx[ptr_object_id] = (object_id_to_idx[ptr_object_id]+1)%len(latent_class_per_obj_list)

    print('saving datasets...')
    filename = os.path.join(dirs,'sqoot.pickle')
    with  open(filename, 'wb') as f:
        pickle.dump((dataset, nX, nb_answers, question_size, idx2answer), f)
    print('datasets saved at {}'.format(filename))

    return dataset, nX, nb_answers, question_size, idx2answer


class SQOOTDataset(Dataset):
    def __init__(self, 
                 root, 
                 img_size,
                 nb_objects,
                 nb_shapes,
                 train_nb_rhs,
                 train=True, 
                 transform=None, 
                 generate=False,
                 random_generation=False,
                 nb_samples=None, 
                 filter_stimulus_on_rhs=False,
                 split_strategy=None,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.5,
                 thickness=1,
                 ):
        super(SQOOTDataset, self).__init__()
        
        self.root = root
        self.file = 'sqoot.pickle'
        self.nb_objects = nb_objects
        self.nb_shapes = nb_shapes
        self.img_size = img_size
        self.font=font
        self.fontScale = fontScale
        self.thickness = thickness

        self.generate = generate
        self.random_generation = random_generation
        self.nb_samples = nb_samples

        self.split_strategy = split_strategy        
        self.transform = transform 

        self.train_nb_rhs = train_nb_rhs
        assert(self.train_nb_rhs < self.nb_shapes)
        self.filter_stimulus_on_rhs = filter_stimulus_on_rhs

        # From lhs to rhs:
        rhss = {
            lhs_shape_id:np.arange(0,self.nb_shapes)
            for lhs_shape_id in range(self.nb_shapes)
        }
        for rhs in rhss.values(): np.random.shuffle(rhs)
        
        nb_testing_rhs = self.nb_shapes-self.train_nb_rhs
        self.training_rhs = {
            lhs_shape_id:possible_rhs[nb_testing_rhs:]
            for lhs_shape_id, possible_rhs in rhss.items()
        }
        self.testing_rhs = {
            lhs_shape_id:possible_rhs[:nb_testing_rhs]
            for lhs_shape_id, possible_rhs in rhss.items()
        }
        
        self.nb_nr_qs=5
        self.nb_r_qs=7
        self.nb_brq_qs=8
                 
        if generate or not self._check_exists():
            if not self._check_exists():
                print('Dataset not found. Let us generate it:')

            dataset, nX, nb_answers, question_size, idx2answer = self._generate(
                root=root,
                nb_r_qs=self.nb_r_qs,
                nb_nr_qs=self.nb_nr_qs,
                nb_brq_qs=self.nb_brq_qs,
                img_size=img_size,
                nb_objects=nb_objects,
                nb_shapes=nb_shapes,
                fontScale=fontScale,
                thickness=thickness,
                random_generation=self.random_generation,
                nb_samples=self.nb_samples
            )
        else:
            filepath = os.path.join(self.root, self.file)
            with open(filepath, 'rb') as f:
              dataset, nX, nb_answers, question_size, idx2answer = pickle.load(f)
        
        self.nX = nX
        self.nb_answers = nb_answers
        self.idx2answer = idx2answer
        self.question_size = question_size
        self.train = train 
        #TODO handle dataset... train test split with combinatorial

        self.latents_values = np.asarray(dataset['latents_values'])
        #(color, shape, X, Y) :
        self.latents_classes = np.asarray(dataset['latents_classes'])
        self.latents_one_hot = np.asarray(dataset['latents_one_hot'])
        
        self.imgs = {} #np.asarray(dataset['imgs'])
        
        self.relational_qs = {idx:{} for idx in range(self.nb_r_qs)}
        # nb_r_qs x |D| x nb_ojects x question_size
        self.relational_as = {idx:{} for idx in range(self.nb_r_qs)}
        # nb_r_qs x |D| x nb_ojects x 1

        self.non_relational_qs = {idx:{} for idx in range(self.nb_nr_qs)}
        # nb_nr_qs x |D| x nb_ojects x question_size
        self.non_relational_as = {idx:{} for idx in range(self.nb_nr_qs)}
        # nb_nr_qs x |D| x nb_ojects x 1

        self.binary_relational_qs = {idx:{} for idx in range(self.nb_brq_qs)}
        # nb_brq_qs x |D| x nb_ojects x nb_shapes x question_size
        self.binary_relational_as = {idx:{} for idx in range(self.nb_brq_qs)}
        # nb_brq_qs x |D| x nb_ojects x nb_shapes x 1

        
        self.targets = np.zeros(len(self.latents_classes))
        for idx, latent_cls in enumerate(self.latents_classes):
            posX = latent_cls[-2]
            posY = latent_cls[-1]
            target = posX*self.nX+posY
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
                # 1: Y
                self.latent_dims['Y'] = {'size': self.nX}
                
                self.latent_dims['Y']['nbr_fillers'] = 0
                self.latent_dims['Y']['primitive'] = ('FP' in strategy[1])
                if self.latent_dims['Y']['primitive']:
                    self.latent_dims['Y']['nbr_fillers'] = int(strategy[1].split('FP')[0])
                self.latent_dims['Y']['image_wise_primitive'] = ('IWP' in strategy[1])
                if self.latent_dims['Y']['image_wise_primitive']:
                    self.latent_dims['Y']['nbr_fillers'] = int(strategy[1].split('IWP')[0])
                    assert  self.latent_dims['Y']['nbr_fillers'] < self.latent_dims['Y']['size']//self.latent_dims['Y']['divider'], \
                            "It seems that the test dataset will be empty."

                self.latent_dims['Y']['position'] = 3
                # (color, shape, X, Y)
                # 2: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                self.latent_dims['Y']['divider'] = int(strategy[2])
                # 3: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.
                if 'N' in strategy[3]:
                    self.latent_dims['Y']['untested'] = True
                    self.latent_dims['Y']['test_set_divider'] = (self.latent_dims['Y']['size']//self.latent_dims['Y']['divider'])+10
                elif 'E' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_end'] = int(strategy[3][1:])
                elif 'S' in strategy[3]:  
                    self.latent_dims['Y']['test_set_size_sample_from_start'] = int(strategy[3][1:])
                else:
                    self.latent_dims['Y']['test_set_divider'] = int(strategy[3])

                # 4: X
                self.latent_dims['X'] = {'size': self.nX}
                
                self.latent_dims['X']['nbr_fillers'] = 0
                self.latent_dims['X']['primitive'] = ('FP' in strategy[4])
                if self.latent_dims['X']['primitive']:
                    self.latent_dims['X']['nbr_fillers'] = int(strategy[4].split('FP')[0])
                self.latent_dims['X']['image_wise_primitive'] = ('IWP' in strategy[4])
                if self.latent_dims['X']['image_wise_primitive']:
                    self.latent_dims['X']['nbr_fillers'] = int(strategy[4].split('IWP')[0])
                    assert  self.latent_dims['X']['nbr_fillers'] < self.latent_dims['X']['size']//self.latent_dims['X']['divider'], \
                            "It seems that the test dataset will be empty."

                self.latent_dims['X']['position'] = 2
                #(color, shape, X, Y)
                # 5: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 8  
                self.latent_dims['X']['divider'] = int(strategy[5])
                # 6: test_set_divider (default:4) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=4 => effective indices 4 and 8 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[6]:
                    self.latent_dims['X']['untested'] = True
                    self.latent_dims['X']['test_set_divider'] = (self.latent_dims['X']['size']//self.latent_dims['X']['divider'])+10
                elif 'E' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_end'] = int(strategy[6][1:])
                elif 'S' in strategy[6]:  
                    self.latent_dims['X']['test_set_size_sample_from_start'] = int(strategy[6][1:])
                else:  
                    self.latent_dims['X']['test_set_divider'] = int(strategy[6])
                # 7: Shape
                self.latent_dims['Shape'] = {'size': self.nb_shapes}
                
                self.latent_dims['Shape']['nbr_fillers'] = 0
                self.latent_dims['Shape']['primitive'] = ('FP' in strategy[7])
                if self.latent_dims['Shape']['primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[7].split('FP')[0])
                self.latent_dims['Shape']['image_wise_primitive'] = ('IWP' in strategy[7])
                if self.latent_dims['Shape']['image_wise_primitive']:
                    self.latent_dims['Shape']['nbr_fillers'] = int(strategy[7].split('IWP')[0])
                
                self.latent_dims['Shape']['position'] = 1
                #(color, shape, X, Y)
                # 8: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=4 => effective size = 10  
                self.latent_dims['Shape']['divider'] = int(strategy[8])
                # 9: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 and 10 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if self.latent_dims['Shape']['image_wise_primitive']:
                    assert  self.latent_dims['Shape']['nbr_fillers'] < self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'], \
                            "It seems that the test dataset will be empty."


                if 'N' in strategy[9]:
                    self.latent_dims['Shape']['untested'] = True
                    self.latent_dims['Shape']['test_set_divider'] = (self.latent_dims['Shape']['size']//self.latent_dims['Shape']['divider'])+10
                elif 'E' in strategy[9]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_end'] = int(strategy[9][1:])
                elif 'S' in strategy[9]:  
                    self.latent_dims['Shape']['test_set_size_sample_from_start'] = int(strategy[9][1:])
                else:  
                    self.latent_dims['Shape']['test_set_divider'] = int(strategy[9])
                '''
                # 10: Color
                self.latent_dims['Color'] = {'size': self.nb_objects}
                
                self.latent_dims['Color']['nbr_fillers'] = 0
                self.latent_dims['Color']['primitive'] = ('FP' in strategy[10])
                if self.latent_dims['Color']['primitive']:
                    self.latent_dims['Color']['nbr_fillers'] = int(strategy[10].split('FP')[0])

                self.latent_dims['Color']['position'] = 0
                #(color, shape, X, Y)
                # 11: divider (default:1) : specify how dense the data are along that dimension
                # e.g. : divider=1 => effective size = 6  
                self.latent_dims['Color']['divider'] = int(strategy[11])
                # 12: test_set_divider (default:5) : out of the effective samples, which indices
                # will be used solely in test, when combined with another latent's test indices.
                # e.g. ~ 80%/20% train/test ==> test_set_divider=5 => effective indices 5 will only be used in the test set,
                # in combination with the other latent dims test set indices.  
                if 'N' in strategy[12]:
                    self.latent_dims['Color']['untested'] = True
                    self.latent_dims['Color']['test_set_divider'] = (self.latent_dims['Color']['size']//self.latent_dims['Color']['divider'])+10
                else:  
                    self.latent_dims['Color']['test_set_divider'] = int(strategy[12])
                '''

                nbr_primitives_and_tested = len([k for k in self.latent_dims 
                    if self.latent_dims[k]['primitive'] \
                    or self.latent_dims[k]['image_wise_primitive'] \
                    or 'untested' not in self.latent_dims[k]])
                assert nbr_primitives_and_tested==self.counter_test_threshold

        #elif 'uniformBinaryRelationalQuery':

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
            indices_latents = list(zip(range(len(self.latents_classes)), self.latents_classes))
            for idx, lc in indices_latents:
                latent_class = lc.reshape((self.nb_objects,-1)).astype(int)
                # (nb_objects, 4)
                effective_test_threshold = self.counter_test_threshold
                #counter_test = {}
                counter_test = np.zeros((self.nb_objects,1))
                # (nb_objects, 1)
                skip_it = False
                filler_forced_training = False
                for dim_name, dim_dict in self.latent_dims.items():
                    dim_class = latent_class[:,dim_dict['position']]
                    # (nb_objects,)
                    quotient = (dim_class+1)//dim_dict['divider']
                    # (nb_objects,)
                    remainder = (dim_class+1)%dim_dict['divider']
                    # (nb_objects,)
                    if any(remainder!=0):
                        skip_it = True
                        break

                    if dim_dict['primitive']:
                        ordinal = quotient
                        # (nb_objects,)
                        if any(ordinal > dim_dict['nbr_fillers']):
                            effective_test_threshold -= 1
                    elif dim_dict['image_wise_primitive']:
                        ordinal = quotient
                        how_many_IWP_values_per_object = (ordinal > dim_dict['nbr_fillers'])
                        # (nb_objects,)
                        how_many_primitive_values_in_image = how_many_IWP_values_per_object.sum()
                        if how_many_primitive_values_in_image >= 2:
                            effective_test_threshold -= 1

                    if 'test_set_divider' in dim_dict: test1 = (quotient%dim_dict['test_set_divider']==0)
                    if 'test_set_divider' in dim_dict and any(test1):
                        counter_test = np.concatenate([counter_test, test1.reshape((-1,1))], axis=1)
                    elif 'test_set_size_sample_from_end' in dim_dict:
                        max_quotient = dim_dict['size']//dim_dict['divider']
                        test2 = (quotient > max_quotient-dim_dict['test_set_size_sample_from_end'])
                        if any(test2):
                            counter_test = np.concatenate([counter_test, test2.reshape((-1,1))], axis=1)
                    elif 'test_set_size_sample_from_start' in dim_dict:
                        test3 = quotient <= dim_dict['test_set_size_sample_from_start']
                        if any(test3):
                            counter_test = np.concatenate([counter_test, test3.reshape((-1,1))], axis=1)

                if skip_it: continue

                if self.train:
                    if any(counter_test.sum(-1) >= effective_test_threshold):
                        continue
                    else:
                        self.indices.append(idx)
                else:
                    if any(counter_test.sum(-1) >= effective_test_threshold):
                        self.indices.append(idx)
                    else:
                        continue

            assert len(self.indices),\
                "No valid data, maybe try a smaller divider..."

            print(f"Split Strategy: {self.split_strategy}")
            print(self.latent_dims)
            print(f"Dataset Size: {len(self.indices)} out of {len(self.latents_values)} : {100*len(self.indices)/len(self.latents_values)}%.")
        

        # Stimulus Filtering:
        # Ensure that stimulus are presented with the allowed set of rhs for each object:
        if self.filter_stimulus_on_rhs:
            raise NotImplementedError
            self.filtered_indices = []
            for idx in self.indices:
                keep = True 

                obj_id2shape_id = [
                    self.latents_classes[idx].reshape(self.nb_objects,-1)[obj_id][1] 
                    for obj_id in range(self.nb_objects)
                ]

                rhs_selection = self.training_rhs
                if not(self.train): rhs_selection = self.testing_rhs
                
                for focus_obj_idx in range(self.nb_objects):
                    allowed_other_shapes = rhs_selection[focus_obj_idx]
                    
                    for other_obj_idx in range(self.nb_objects):
                        if focus_obj_idx == other_obj_idx: continue
                        if obj_id2shape_id[other_obj_idx] not in allowed_other_shapes:
                            keep = False
                            break

                    if not(keep): break

                if keep:
                    self.filtered_indices.append(idx)

            self.indices = self.filtered_indices
                

        self.latents_values = self.latents_values[self.indices]
        self.latents_classes = self.latents_classes[self.indices]
        self.latents_one_hot = self.latents_one_hot[self.indices]

        """
        self.imgs = self.imgs[self.indices]
        
        self.relational_qs = {k:v[self.indices] for k,v in self.relational_qs.items()}
        self.relational_as = {k:v[self.indices] for k,v in self.relational_as.items()}
        
        self.non_relational_qs = {k:v[self.indices] for k,v in self.non_relational_qs.items()}
        self.non_relational_as = {k:v[self.indices] for k,v in self.non_relational_as.items()}
        
        self.binary_relational_qs = {k:v[self.indices] for k,v in self.binary_relational_qs.items()}
        self.binary_relational_as = {k:v[self.indices] for k,v in self.binary_relational_as.items()}
        """

        self.targets = self.targets[self.indices]

        print('Dataset loaded : OK.')
    
    def _generate_all(self):
        pbar = tqdm(total=len(self))
        for idx in range(len(self)):
            pbar.update(1)
            self._generate_datapoint(idx=idx)

    def _generate_datapoint(self, idx):
        latents_values = self.latents_values[idx].reshape(self.nb_objects, -1)
        latents_one_hot = self.latents_one_hot[idx].reshape(self.nb_objects, -1)
        latents_classes = self.latents_classes[idx].reshape(self.nb_objects, -1)
        
        datapoint = generate_datapoint(
            latent_one_hot=latents_one_hot, 
            latent_values=latents_values,
            latent_classes=latents_classes,
            nb_r_qs=self.nb_r_qs,
            nb_nr_qs=self.nb_nr_qs,
            nb_brq_qs=self.nb_brq_qs,
            img_size=self.img_size,
            nb_objects=self.nb_objects,
            nb_shapes=self.nb_shapes,
            font=self.font,
            fontScale=self.fontScale,
            thickness=self.thickness,
        )

        #(img, relations, norelations, latent_class.reshape(-1), latent_values.reshape(-1))
        self.imgs[idx] = datapoint[0]
        
        for subtype_id in range(self.nb_r_qs):
            self.relational_qs[subtype_id][idx] = np.stack(datapoint[1][0][subtype_id])
            # nb_r_qs x |D| x nb_ojects x question_size
            self.relational_as[subtype_id][idx] = np.stack(datapoint[1][1][subtype_id])
            # nb_r_qs x |D| x nb_ojects x 1

            """
            dataset[f"relational_qs_{subtype_id}"].append(np.stack(datapoint[1][0][subtype_id]))
            # nb_brq_qs x nb_ojects x  question_size
            dataset[f"relational_as_{subtype_id}"].append(np.stack(datapoint[1][1][subtype_id]))
            # nb_r_qs x nb_ojects x 1
            """

        for subtype_id in range(self.nb_nr_qs):
            self.non_relational_qs[subtype_id][idx] = np.stack(datapoint[2][0][subtype_id])
            # nb_r_qs x |D| x nb_ojects x question_size
            self.non_relational_as[subtype_id][idx] = np.stack(datapoint[2][1][subtype_id])
            # nb_r_qs x |D| x nb_ojects x 1

            """
            dataset[f"non_relational_qs_{subtype_id}"].append(np.stack(datapoint[2][0][subtype_id]))
            # nb_brq_qs x nb_ojects x question_size
            dataset[f"non_relational_as_{subtype_id}"].append(np.stack(datapoint[2][1][subtype_id]))
            # nb_nr_qs x nb_ojects x 1
            """

        for subtype_id in range(self.nb_brq_qs):
            self.binary_relational_qs[subtype_id][idx] = np.stack(datapoint[3][0][subtype_id])
            # nb_r_qs x |D| x nb_ojects x question_size
            self.binary_relational_as[subtype_id][idx] = np.stack(datapoint[3][1][subtype_id])
            # nb_r_qs x |D| x nb_ojects x 1

            """
            dataset[f"binary_relational_query_qs_{subtype_id}"].append(np.stack(datapoint[3][0][subtype_id]))
            # nb_brq_qs x nb_ojects x nb_shapes x question_size
            dataset[f"binary_relational_query_as_{subtype_id}"].append(np.stack(datapoint[3][1][subtype_id]))
            # nb_brq_qs x nb_ojects x nb_shapes x 1
            """

        #self.relational_qs = {idx:np.stack(dataset[f'relational_qs_{idx}']) for idx in range(self.nb_r_qs)}
        # nb_r_qs x |D| x nb_ojects x question_size
        #self.relational_as = {idx:np.stack(dataset[f'relational_as_{idx}']) for idx in range(self.nb_r_qs)}
        # nb_r_qs x |D| x nb_ojects x 1

        #self.non_relational_qs = {idx:np.stack(dataset[f'non_relational_qs_{idx}']) for idx in range(self.nb_nr_qs)}
        # nb_nr_qs x |D| x nb_ojects x question_size
        #self.non_relational_as = {idx:np.stack(dataset[f'non_relational_as_{idx}']) for idx in range(self.nb_nr_qs)}
        # nb_nr_qs x |D| x nb_ojects x 1

        #self.binary_relational_qs = {idx:np.stack(dataset[f'binary_relational_query_qs_{idx}']) for idx in range(self.nb_brq_qs)}
        # nb_brq_qs x |D| x nb_ojects x nb_shapes x question_size
        #self.binary_relational_as = {idx:np.stack(dataset[f'binary_relational_query_as_{idx}']) for idx in range(self.nb_brq_qs)}
        # nb_brq_qs x |D| x nb_ojects x nb_shapes x 1

    def __len__(self) -> int:
        return len(self.indices)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,self.file))

    def _generate(self, 
                  root,
                  nb_r_qs,
                  nb_nr_qs,
                  nb_brq_qs,
                  img_size,
                  nb_objects,
                  nb_shapes,
                  fontScale=0.5,
                  thickness=1,
                  random_generation=False,
                  nb_samples=None):
        """
        Generate the SQOOT dataset if it doesn't exist already.
        """
        if root is None:
            root = self.root
        os.makedirs(root, exist_ok=True)
        return generate_dataset(
            root=root,
            nb_r_qs=nb_r_qs,
            nb_nr_qs=nb_nr_qs,
            nb_brq_qs=nb_brq_qs,
            img_size=img_size,
            nb_objects=nb_objects,
            nb_shapes=nb_shapes,
            fontScale=fontScale,
            thickness=thickness,
            random_generation=random_generation,
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

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if idx >= len(self):
            idx = idx%len(self)

        latent_value = torch.from_numpy(self.getlatentvalue(idx))
        latent_class = torch.from_numpy(self.getlatentclass(idx))
        latent_one_hot = torch.from_numpy(self.getlatentonehot(idx))
        
        if idx not in self.imgs:    
            self._generate_datapoint(idx=idx)

        img = self.imgs[idx]
        target = self.getclass(idx)
                
        relational_questions = {f"relational_questions_{k}":torch.from_numpy(v[idx]).float() for k,v in self.relational_qs.items()}
        relational_answers = {f"relational_answers_{k}":torch.from_numpy(v[idx]).long() for k,v in self.relational_as.items()}
        
        non_relational_questions = {f"non_relational_questions_{k}":torch.from_numpy(v[idx]).float() for k,v in self.non_relational_qs.items()}
        non_relational_answers = {f"non_relational_answers_{k}":torch.from_numpy(v[idx]).long() for k,v in self.non_relational_as.items()}
        
        binary_relational_questions = {f"binary_relational_query_questions_{k}":torch.from_numpy(v[idx]).float() for k,v in self.binary_relational_qs.items()}
        binary_relational_answers = {f"binary_relational_query_answers_{k}":torch.from_numpy(v[idx]).long() for k,v in self.binary_relational_as.items()}
        
        # Filtering #RHS/#LHS:
        for (key, vqs), (akey,vas) in zip(binary_relational_questions.items(), binary_relational_answers.items()):
            # nb_objects x nb_shapes x question_size
            obj_id2shape_id = [
                latent_class.reshape(self.nb_objects,-1)[obj_id][1].item() 
                for obj_id in range(self.nb_objects)
            ]
            rhs_selection = self.training_rhs
            if not(self.train): rhs_selection = self.testing_rhs
            selected_qs = []
            selected_as = []
            for object_id in range(self.nb_objects):
                objXshape = torch.from_numpy(rhs_selection[obj_id2shape_id[object_id]])
                selected_qs.append(torch.index_select(
                        vqs[object_id],
                        dim=0,
                        index=objXshape
                    )
                )
                selected_as.append(torch.index_select(
                        vas[object_id],
                        dim=0,
                        index=objXshape
                    )
                )
            binary_relational_questions[key] = torch.stack(selected_qs)
            binary_relational_answers[akey] = torch.stack(selected_as)

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
            "exp_latents_one_hot_encoded":latent_one_hot
        }
        
        sampled_d.update(relational_questions)
        sampled_d.update(relational_answers)
        
        sampled_d.update(non_relational_questions)
        sampled_d.update(non_relational_answers)

        sampled_d.update(binary_relational_questions)
        sampled_d.update(binary_relational_answers)
        
        return sampled_d