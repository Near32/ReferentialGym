from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions.categorical import Categorical

import numpy as np 

from .module import Module


def build_CoOccurrenceSemanticGroundingLossModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None,
) -> Module:
    
    return CoOccurrenceSemanticGroundingLossModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class CoOccurrenceSemanticGroundingLossModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object]={
            "lambda_factor":1.0,
            "noise_magnitude":0.0,
        },
        input_stream_ids:Dict[str,str]=None,
    ):
        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "mode":"signals:mode",

            "agent":"modules:current_speaker:ref:ref_agent",
            "visual_features":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:visual_features",
            "text_features":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:text_features",
            "semantic_prior":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:semantic_prior",
            "semantic_prior_logits":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:semantic_prior_logits",
            "grounding_signal":"current_dataloader:sample:speaker_grounding_signal",
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_stream, default_id in default_input_stream_ids.items():
                if default_id not in input_stream_ids.values():
                    input_stream_ids[default_stream] = default_id

        super(CoOccurrenceSemanticGroundingLossModule, self).__init__(
            id=id,
            type="CoOccurrenceSemanticGroundingLossModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.noise_magnitude = self.config.get('noise_magnitude', 0.0)
        self.noisy = self.noise_magnitude > 1.0e-3
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        losses_dict = input_streams_dict["losses_dict"]
        it_rep = input_streams_dict["it_rep"]
        it_comm_round = input_streams_dict["it_comm_round"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        agent = input_streams_dict["agent"]
        
        visual_features = input_streams_dict['visual_features']
        # (batch_size x nbr_visual_emb = 1 x emb_size)
        batch_size = visual_features.shape[0]
        nbr_visual_features = visual_features.shape[1]
        text_features = input_streams_dict['text_features']
        # (batch_size x nbr_text_emb = vocab_size x emb_size)
        nbr_text_features = text_features.shape[1]
        semantic_prior_logits = input_streams_dict["semantic_prior_logits"]
        # (batch_size x nbr_visual_emb =1 x nbr_text_emb=vocab_size)
        semantic_prior = input_streams_dict["semantic_prior"]
        # (batch_size x nbr_text_emb=vocab_size)
        sem_prior_distr = Categorical(probs=semantic_prior)
        entropy = sem_prior_distr.entropy()

        grounding_signal = input_streams_dict["grounding_signal"].squeeze(-1)
        max_sentence_length = grounding_signal.shape[-1]
        grounding_signal = grounding_signal.reshape((batch_size, max_sentence_length))
        # (batch_size x max_sentence_length)
        
        # Compute Loss Function: 
        noise = 0.0
        targets_logits = (-1)*torch.ones((batch_size, nbr_text_features, nbr_visual_features))
        for tfidx in range(nbr_text_features):
            for vfidx in range(nbr_visual_features):
                tfidx_mask = (grounding_signal == tfidx)
                tfidx_indices = torch.nonzero(tfidx_mask.sum(-1), as_tuple=True)
                target = 1.0
                if len(tfidx_indices[0]) == batch_size:
                    # filtering out the text features with zero entropy
                    # TODO: investigate whether putting a null target could be beneficial?
                    continue
                elif len(tfidx_indices[0]) == 0 :   continue
                if self.noisy:  noise = torch.rand(1).item()*self.noise_magnitude
                targets_logits[tfidx_indices[0], tfidx, vfidx] = target-noise
        
        targets_logits = targets_logits.to(semantic_prior.device)
        loss = torch.square(semantic_prior_logits-targets_logits).mean(-1).mean(-1)
        # (batch_size, )
        
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Entropy"] = entropy
        losses_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Loss"] = [self.config.get("lambda_factor", 1.0), loss]

        return outputs_stream_dict
    
