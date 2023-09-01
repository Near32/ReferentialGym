from typing import Dict, List 

import torch
import torch.nn as nn

import numpy as np 

from .module import Module


def build_OrthogonalityMetricModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None,
) -> Module:
    
    return OrthogonalityMetricModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class OrthogonalityMetricModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object]={},
        input_stream_ids:Dict[str,str]=None,
    ):
        '''
        '''
        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "mode":"signals:mode",

            "agent":"modules:current_speaker:ref:ref_agent",
            "representations":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:semantic_embedding:weight",
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_stream, default_id in default_input_stream_ids.items():
                if default_id not in input_stream_ids.values():
                    input_stream_ids[default_stream] = default_id

        super(OrthogonalityMetricModule, self).__init__(
            id=id,
            type="OrthogonalityMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
    
    def _compute_ortho_dist(self, A):
        with torch.no_grad():
            q,r = torch.linalg.qr(A)
            eye = torch.eye(r.shape[0], r.shape[1]).to(r.device)
            diff = r-eye
            dist = diff.norm(p=2).item()
        return dist

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
        representations = input_streams_dict["representations"]
        
        ortho_dist = self._compute_ortho_dist(representations)
        
        name = '/'.join(self.input_stream_ids['representations'].split(':')[-3:])
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.id}/{agent.agent_id}/{name}"] = ortho_dist

        return outputs_stream_dict
 
