from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module

def build_GradRecorderModule(id:str,
                               config:Dict[str,object]=None,
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return GradRecorderModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class GradRecorderModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        input_stream_ids = {
            "logs_dict":"logs_dict",
            "signals:mode":"mode",
            "signals:it_sample":"it_sample",
            # step in the sequence of repetitions of the current batch
            "signals:it_step":"it_step",
            # step in the communication round.
            "modules:current_speaker:ref:ref_agent":"current_speaker",
            "modules:current_listener:ref:ref_agent":"current_listener",
        }

        super(GradRecorderModule, self).__init__(id=id,
                                                 type="GradRecorderModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        '''
        '''
        outputs_stream_dict = {}

        logs_dict = input_streams_dict['logs_dict']
        
        mode = input_streams_dict['mode']

        if 'train' in mode:
            it_rep = input_streams_dict['it_sample']
            it_comm_round = input_streams_dict['it_step']
            
            speaker = input_streams_dict['current_speaker']
            listener = input_streams_dict['current_listener']
            
            maxgrad = 0.0
            for name, p in speaker.named_parameters() :
                if hasattr(p,'grad') and p.grad is not None:
                    logs_dict[f'{mode}/repetition{it_rep}/comm_round{it_comm_round}/current_speaker/grad/{name}'] = p.grad.cpu().detach()
                    cmg = torch.abs(p.grad.cpu().detach()).max()
                    if cmg > maxgrad:
                        maxgrad = cmg
            logs_dict[f'{mode}/repetition{it_rep}/comm_round{it_comm_round}/current_speaker/max_grad'] = maxgrad
            
            maxgrad = 0.0
            for name, p in listener.named_parameters() :
                if hasattr(p,'grad') and p.grad is not None:
                    logs_dict[f'{mode}/repetition{it_rep}/comm_round{it_comm_round}/current_listener/grad/{name}'] = p.grad.cpu().detach()
                    cmg = torch.abs(p.grad.cpu().detach()).max()
                    if cmg > maxgrad:
                        maxgrad = cmg
            logs_dict[f'{mode}/repetition{it_rep}/comm_round{it_comm_round}/current_listener/max_grad'] = maxgrad
            
        return outputs_stream_dict
    