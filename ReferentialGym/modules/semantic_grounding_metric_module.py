from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions.categorical import Categorical

import numpy as np 

from .module import Module


def build_SemanticGroundingMetricModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None,
) -> Module:
    
    return SemanticGroundingMetricModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class SemanticGroundingMetricModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object]={},
        input_stream_ids:Dict[str,str]=None,
    ):
        '''
        :param config:
            - 'idx2w' : Dict[int, str] describing the vocabulary used by the agent.
            - 'semantic_percentiles': List[float] between 0 and 100 inclusives.
        '''
        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "mode":"signals:mode",

            "agent":"modules:current_speaker:ref:ref_agent",
            "sentences":"modules:current_speaker:sentences_widx",
            "semantic_signal":"current_dataloader:sample:speaker_semantic_signal",
            "semantic_prior":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:semantic_prior",
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_stream, default_id in default_input_stream_ids.items():
                if default_id not in input_stream_ids.values():
                    input_stream_ids[default_stream] = default_id

        super(SemanticGroundingMetricModule, self).__init__(
            id=id,
            type="SemanticGroundingMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.idx2w = self.config['idx2w']
        self.w2idx = dict(zip(self.idx2w.values(), self.idx2w.keys()))
        self.semantic_percentiles = self.config['semantic_percentiles']
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        it_rep = input_streams_dict["it_rep"]
        it_comm_round = input_streams_dict["it_comm_round"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        agent = input_streams_dict["agent"]
        
        sentences = input_streams_dict['sentences']
        # (batch_size x max_sentence_length x 1)
        batch_size = sentences.shape[0]
        max_sentence_length = sentences.shape[-1]
        semantic_signal = input_streams_dict['semantic_signal']
        # (batch_size x nbr_stimuli=1 x nbr_distractors_po=1 x **semantic_dims)
        semantic_prior = input_streams_dict['semantic_prior']
        # (batch_size x vocab_size)
	 
        idx2w = self.idx2w
        COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
        IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "door": 4,
            "key": 5,
            "ball": 6,
            "box": 7,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }
        IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
        
        sentences_w = [[self.idx2w[token.item()] for token in sentence] for sentence in sentences] 
        symb_image = semantic_signal.squeeze().cpu()
        accuracies = {
            k: {'nbr_success':0, 'nbr_occ':0, 'occs':None}
            for k in [
                'any-shape', # Is any of the visible shapes mentioned?
                'all-shape', # Are all the visible shapes mentioned, and none more?
                'any-color', # Is any of the visible colors mentioned?
                'all-color', # Are all the visible colors mentioned, and none more?
                'any-object', # Is any of the visible object mentioned?
                'all-object', # Are all the visible objects mentioned, and none more?
            ]
        }
        
        percentiles = None
        if semantic_prior is not None:
            domains = ['color', 'shape', 'object']
            types = ['any', 'all']
            for pidx in range(len(self.semantic_percentiles)):
                for t in types:
                    for d in domains:
                        accuracies[f"sem-{t}-{d}-{pidx}"] = {
                            'nbr_success':0,
                            'nbr_occ':0,
                            'occs': None
                        }
            
            percentiles = np.percentile(
                a=semantic_prior.cpu().detach().numpy(),
                q=self.semantic_percentiles,
                axis=-1,
                keepdims=False,
            )
        
        # (pidx x batch_size)
        for bidx in range(batch_size):
            visible_shapes = []
            visible_colors = []
            visible_objects = []
            for i in range(symb_image.shape[1]):
                for j in range(symb_image.shape[2]):
                    if symb_image[bidx,i,j,0] <= 3 : continue
                    color_idx = symb_image[bidx,i,j,1].item()
                    shape_idx = symb_image[bidx,i,j,0].item()
                    color = IDX_TO_COLOR[color_idx]
                    shape = IDX_TO_OBJECT[shape_idx]
                
                    visible_shapes.append(shape)
                    visible_colors.append(color)
                    visible_objects.append((color,shape))
            d2v = {
                'shape':visible_shapes,
                'color':visible_colors,
                'object':visible_objects,
            }
            for k in accuracies:
                if 'object' in k:  continue
                splits = k.split('-')
                if 'sem' in splits:
                    acc_type = splits[1]
                    acc_domain = splits[2]
                    percentile_idx = int(splits[3])
                else:
                    acc_type = splits[0]
                    acc_domain = splits[1]
                if acc_type=='any':
                    filter_fn = any
                else:
                    filter_fn = all
                
                occs = {}
                for sem in d2v[acc_domain]:
                    if sem in occs \
                    and occs[sem]==1:
                        continue
                    occs[sem]=0
                    if 'sem' in splits:
                        # occs[sem] = 1 if and only if the likelihood
                        # of the token is above the pidx-th percentile:
                        threshold_prob = percentiles[percentile_idx, bidx]
                        token_idx = self.w2idx[sem]
                        token_prob = semantic_prior[bidx, token_idx]
                        if token_prob.item() >= threshold_prob:
                            occs[sem] = 1
                    else:
                        for word in sentences_w[bidx]:
                            if word==sem:
                                occs[sem]=1
                                break
                # WARNING: all([]) -> True, which defies the purpose...
                if len(occs) == 0:
                    acc = 0
                else:
                    acc = filter_fn(occs.values())
                accuracies[k]['occs'] = occs
                accuracies[k]['nbr_success'] += int(acc)
                if len(d2v[acc_domain]):
                    accuracies[k]['nbr_occ'] += 1
            # Need to compute it for each parts before computing for objects as whole:
            for k in accuracies:
                if 'object' not in k:   
                    continue
                splits = k.split('-')
                if 'sem' in splits:
                    acc_type = splits[1]
                    acc_domain = splits[2]
                    pidx = splits[3]
                    color_k = f"sem-{acc_type}-color-{pidx}"
                    shape_k = f"sem-{acc_type}-shape-{pidx}"
                else:
                    acc_type = splits[0]
                    acc_domain = splits[1]
                    color_k = f"{acc_type}-color"
                    shape_k = f"{acc_type}-shape"
                if acc_type=='any':
                    filter_fn = any
                else:
                    filter_fn = all
                occs = [all([
                    accuracies[color_k]['occs'][color],
                    accuracies[shape_k]['occs'][shape],
                    ])
                    for color,shape in visible_objects
                ]
                # WARNING: all([]) -> True, which defies the purpose...
                if len(occs) == 0:
                    acc = 0
                else:
                    acc = filter_fn(occs)
                accuracies[k]['occs'] = occs
                accuracies[k]['nbr_success'] += int(acc)
                if len(d2v[acc_domain]):
                    accuracies[k]['nbr_occ'] += 1
        
        for k in accuracies:
            accuracies[k]['accuracy'] = float(accuracies[k]['nbr_success'])/(1.0e-4+accuracies[k]['nbr_occ'])*100.0
            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.id}/{agent.agent_id}/NbrOcc-{k}"] = accuracies[k]['nbr_occ']
            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.id}/{agent.agent_id}/NbrSucc-{k}"] = accuracies[k]['nbr_success']
            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.id}/{agent.agent_id}/Accuracy-{k}"] = accuracies[k]['accuracy']

        return outputs_stream_dict
 
