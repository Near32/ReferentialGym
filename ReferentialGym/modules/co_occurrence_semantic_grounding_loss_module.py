from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

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
            "semantic_prior":"modules:current_speaker:ref:ref_agent:semantic_prior",
            "grounding_signal":"current_dataset:samples:grounding_signal_widx",
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

        semantic_prior = input_streams_dict["semantic_prior"]
        # (batch_size x vocab_size)
        sem_prior_distr = Categorical(probs=semantic_prior)
        entropy = sem_prior_distr.entropy()

        grounding_signal = input_streams_dict["grounding_signal"]
        # (batch_size x max_sentence_length x 1)
        
        # Compute Loss Function: 
        criterion = nn.MSELoss(reduction='none')
        projection_probs = embedded_sentences_features_mask_logits.softmax(-1)
        # (batch_size, feature_size, max_sentence_length)
        max_projection_probs, argmax_projection_probs = projection_probs.max(dim=-1)
        # (batch_size, feature_size)

        loss = criterion( 
            (max_projection_probs*features), 
            features
        ).mean(dim=-1)
        # mean over each feature dimension...
        # (batch_size, )
        # This loss can be minimized by reducing the feature values to 0 (prior)
        # or by making sure that the symbols projection are disentangled on said feature dimension,
        # i.e. that the symbols are projecting in a disentangled fashion.
        
        logs_dict[f"repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Entropy"] = entropy
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Loss"] = [self.config.get("lambda_factor", 1.0), loss]

        return outputs_stream_dict
    
