from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions.categorical import Categorical

import numpy as np 

from .module import Module

import wandb


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
            "sentence_level_lambda_factor":1.0,
            "noise_magnitude":0.0,
            "semantic_level_grounding": False,
            "semantic_level_ungrounding": False,
            "sentence_level_grounding": True,
            "sentence_level_ungrounding": False,
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
            "sentences_widx":"modules:current_speaker:sentences_widx",
            "sentences_logits":"modules:current_speaker:sentences_logits",
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

        self.semantic_level_grounding = self.config.get('semantic_level_grounding', False)
        self.semantic_level_ungrounding = self.config.get('semantic_level_ungrounding', False)
        self.sentence_level_grounding = self.config.get('sentence_level_grounding', False)
        self.sentence_level_ungrounding = self.config.get('sentence_level_ungrounding', False)
        
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
        sentences_logits = input_streams_dict["sentences_logits"]
        # (batch_size x max_sentence_lengths x vocab_size)
    
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
        mask = torch.zeros((batch_size, nbr_text_features, nbr_visual_features))
        # Sentence level :
        sentences_targets_logits = torch.zeros((batch_size, nbr_text_features))
        sentences_mask = torch.zeros((batch_size, nbr_text_features))
        # Eos Filtering:
        eos_idx = agent.vocab_stop_idx
        sentences_logits[..., eos_idx] = sentences_logits.min(dim=-1, keepdim=False)[0]
        # Aggregation can be done in many different ways:
        #sentences_mean_logits = torch.mean(sentences_logits, dim=1, keepdim=False)
        sentences_mean_logits, _ = torch.max(sentences_logits, dim=1, keepdim=False)
        # (batch_size x vocab_size)
        sentences_mean_probs=sentences_mean_logits.softmax(dim=-1)
        sentences_distr = Categorical(probs=sentences_mean_probs)
        sentence_level_entropy = sentences_distr.entropy()
         
        #histogram_tfidx = []
        for tfidx in range(nbr_text_features):
            tfidx_mask = (grounding_signal == tfidx)
            nontfidx_mask = (grounding_signal != tfidx)
            tfidx_indices = torch.nonzero(tfidx_mask.sum(-1), as_tuple=True)
            nontfidx_indices = torch.nonzero(nontfidx_mask.sum(-1), as_tuple=True)
                
            # Address values when entropy over batch is non null:
            # i.e. either something positive or negative occurs.
            # Otherwise, they are masked out of the loss...
            
            if len(tfidx_indices[0]) == batch_size:
                # filtering out the text features with zero entropy
                continue
                #
                #histogram_tfidx.append(tfidx)
                # 
                # After examination, it turns out that the only words
                # with zero entropy are indeed words that we do not want
                # to ground, because either already grounded (e.g. EoS token)
                # or not really helping (e.g. 'pick up' from a PickUpDist task).
            
            if self.semantic_level_grounding:
                for vfidx in range(nbr_visual_features):
                    if self.noisy:  noise = torch.rand(1).item()*self.noise_magnitude
                    postarget = 1.0-noise
                    negtarget = -1.0+noise
                    
                    if len(tfidx_indices[0]):
                        mask[tfidx_indices[0], tfidx, vfidx] = 1.0
                        targets_logits[tfidx_indices[0], tfidx, vfidx] = postarget
                    if self.semantic_level_ungrounding \
                    and len(nontfidx_indices[0]):
                        mask[nontfidx_indices[0], tfidx, vfidx] = 1.0
                        targets_logits[nontfidx_indices[0], tfidx, vfidx] = negtarget
            
            # Sentence level :
            if self.sentence_level_grounding:
                if self.noisy:    noise = torch.rand(1).item()*self.noise_magnitude
                postarget = 1.0-noise
                negtarget = noise
                sentences_targets_logits[tfidx_indices[0], tfidx] = postarget
                sentences_mask[tfidx_indices[0], tfidx] = 1.0
                if self.sentence_level_ungrounding:
                    sentences_targets_logits[nontfidx_indices[0], tfidx] = negtarget
                    sentences_mask[nontfidx_indices[0], tfidx] = 1.0
 
        #wandb.log({f"{mode}/co_occurrence_semantic_grounding/{agent.agent_id}/TokenIdxFullBatch":wandb.Histogram(histogram_tfidx),}, commit=False)
        if self.semantic_level_grounding:
            targets_logits = targets_logits.to(semantic_prior.device)
            mask = mask.to(semantic_prior.device)
            loss = (mask*torch.square(semantic_prior_logits-targets_logits)).mean(-1).mean(-1)
        else:
            loss = torch.zeros_like(semantic_prior).sum(dim=-1)
        # (batch_size, )
        
        # Sentence level :
        if self.sentence_level_grounding:
            sentences_targets_logits = sentences_targets_logits.to(sentences_mean_probs.device)
            sentences_mask = sentences_mask.to(sentences_mean_probs.device)
            sentences_loss = (sentences_mask*torch.square(sentences_targets_logits-sentences_mean_probs)).mean(-1)
        else:
            sentences_loss = torch.zeros_like(loss)
        # (batch_size, )
        
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Entropy"] = entropy
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/SentenceLevelEntropy"] = sentence_level_entropy
        losses_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/Loss"] = [self.config.get("lambda_factor", 1.0), loss]
        losses_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/co_occurrence_semantic_grounding/{agent.agent_id}/SentenceLevelLoss"] = [self.config.get("sentence_level_lambda_factor", 1.0), sentences_loss]

        return outputs_stream_dict
    
