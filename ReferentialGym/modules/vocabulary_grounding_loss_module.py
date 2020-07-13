from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module
from ..utils import PositionalEncoding
from ..networks import choose_architecture

def build_VocabularyGroundingLossModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    
    features_masks_scorer = choose_architecture(
        architecture=config["architecture"],
        kwargs=config,
        input_shape=config['symbol_embedding_size'],
        feature_dim=config["feature_dim"],
        fc_hidden_units_list=config["fc_hidden_units"],
        dropout=config["dropout_prob"],
    )

    return VocabularyGroundingLossModule(
        id=id,
        features_masks_scorer=features_masks_scorer,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class VocabularyGroundingLossModule(Module):
    def __init__(self,
                 id:str,
                 features_masks_scorer:nn.Module,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "mode":"signals:mode",

            "agent":"modules:current_listener:ref:ref_agent",
            "features":"modules:current_speaker:ref:ref_agent:features",
            "sentences_logits":"modules:current_speaker:sentences_logits",
            "sentences_one_hot":"modules:current_speaker:sentences_one_hot",
            "sentences_widx":"modules:current_speaker:sentences_widx", 
               
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_stream, default_id in default_input_stream_ids.items():
                if default_id not in input_stream_ids.values():
                    input_stream_ids[default_stream] = default_id

        super(VocabularyGroundingLossModule, self).__init__(
            id=id,
            type="VocabularyGroundingLossModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.positional_encoder = PositionalEncoding(
            d_model=config['symbol_embedding_size'],
            dropout=config['positional_encoder_dropout'],
            max_len=config['max_sentence_length']
        )

        self.features_masks_scorer = features_masks_scorer 

        if config["use_cuda"]:
            self.features_masks_scorer = self.features_masks_scorer.cuda() 


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

        features = input_streams_dict["features"]
        # (batch_size, nbr_distractor_po, nbr_stimulus, feature_size)
        # Assume partial: 
        feature_size = features.shape[-1]
        features = features.reshape(-1, feature_size)
        
        # Embed sentences:
        if agent.use_sentences_one_hot_vectors:
            sentences = input_streams_dict["sentences_one_hot"]
        else:
            sentences = input_streams_dict["sentences_widx"]
        embedded_sentences = agent.embed_sentences(sentences) 
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        embedded_sentences_pos = self.positional_encoder(embedded_sentences)
        
        batch_size = embedded_sentences_pos.shape[0]
        max_sentence_length = embedded_sentences_pos.shape[1]
        symbol_embedding_size = embedded_sentences_pos.shape[2]
        
        # Compute Masks Scores:
        embedded_sentences_pos = embedded_sentences_pos.reshape((-1, symbol_embedding_size))
        embedded_sentences_features_scores = self.features_masks_scorer(embedded_sentences_pos)
        # (batch_size*max_sentence_length, feature_size)

        # Compute Masks:
        embedded_sentences_features_mask_logits = embedded_sentences_features_scores.reshape(
            (batch_size, -1, feature_size)
        ).transpose(1,2)
        # (batch_size, feature_size, max_sentence_length)
        
        # Compute Loss Function: 
        """
        # Winner Takes All Loss: does not affect the features, only the embeddings...
        # For each feature, we select the symbol with the highest probability as the target:
        target = embedded_sentences_features_mask_logits.argmax(-1).detach()
        # (batch_size, feature_size)
        
        criterion = nn.NLLLoss(reduction="none")
        masks_logits = F.log_softmax( embedded_sentences_features_mask_logits, dim=-1)
        # (batch_size, feature_size, max_sentence_length)
        loss = criterion( 
            masks_logits.reshape(-1, max_sentence_length), 
            target.reshape(-1)
        ).reshape((batch_size, feature_size)).sum(-1)
        # Sum over each features...
        # (batch_size, )
        
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/vocabulary_grounding_loss/winner_takes_all"] = [1.0, loss]
        """

        # Projection Loss: 
        """
        For each feature dimension, it furthers the symbols with the maximum projection probability 
        to fully explain the feature value at the given dimension.
        Thus, it pushes the features towards the embeddings and vice-versa.
        """
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
        
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/vocabulary_grounding/{agent.agent_id}/projection_loss"] = [1.0, loss]

        # EoS Projection to Null:
        """
        Further the EoS symbol to have its projection probability at zeros, for every feature dimension.
        """
        sentences_widx = input_streams_dict["sentences_widx"]
        eos_mask = (sentences_widx == agent.vocab_stop_idx).reshape((batch_size, 1, -1))
        # (batch_size, 1, max_sentence_length)
        eos_mask = eos_mask.expand(-1, feature_size, -1)
        
        # sum over the symbol dimension since there can only be one EoS symobl per sentence...
        eos_masked_projection_probs = (eos_mask*projection_probs).sum(-1)
        # (batch_size, feature_size)
        
        target_probs = torch.zeros_like(eos_masked_projection_probs)
        criterion = nn.MSELoss(reduction='none')
        eos_loss = criterion(
            eos_masked_projection_probs,
            target_probs
        ).sum(-1)
        # sum over each features...
        # (batch_size, )

        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/vocabulary_grounding/{agent.agent_id}/eos_regularization_loss"] = [1.0, loss]

        return outputs_stream_dict
    
