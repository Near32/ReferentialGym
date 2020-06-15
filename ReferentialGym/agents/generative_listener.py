from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 
import random 

import copy

from .listener import Listener


def generative_st_gs_referential_game_loss(agent,
                                           losses_dict,
                                           input_streams_dict,
                                           outputs_dict,
                                           logs_dict,
                                           **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]
    mode = input_streams_dict["mode"]

    generative_output = outputs_dict["generative_output"]
    # (batch_size, seq_length(==1), depth_dim, width_dim, height_dim )
    batch_size = generative_output.shape[0]


    target_output = input_streams_dict["target_output"]
    if target_output is None:
        sample = input_streams_dict["sample"]
        target_output = sample["speaker_experiences"]

    target_output = target_output.reshape(batch_size,-1)

    if config["agent_loss_type"].lower() == "bce":
        # Reconstruction loss :
        generative_output = generative_output.reshape(batch_size,-1)
        loss = F.binary_cross_entropy_with_logits(
            generative_output,
            target_output,
            reduction="none").mean(-1)
        # (batch_size, )
        neg_log_lik = -torch.distributions.Bernoulli( 
            torch.sigmoid(generative_output) 
        ).log_prob(
            target_output
        )
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/generative_referential_game_neg_log_lik"] = neg_log_lik

        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, loss]
    
    elif config["agent_loss_type"].lower() == "ce":
        # Reconstruction loss :
        assert(len(generative_output.shape)==3)
        
        target_output = input_streams_dict["target_output"].reshape(batch_size,-1)
        
        n_dim = target_output.shape[-1]
        n_classes = generative_output.shape[-1]//n_dim
        
        generative_output = generative_output.reshape(batch_size,n_classes, n_dim)
        
        losses = F.cross_entropy(
            generative_output,
            target_output.long(),
            reduction="none")
        # (batch_size, n_dim)
        loss = losses.mean(-1)
        # (batch_size, )
        
        # Accuracy:
        argmax_generative_output = generative_output.argmax(dim=1)
        # (batch_size, n_dim)
        accuracies = 100.0*(target_output==argmax_generative_output).float()
        # (batch_size, n_dim)
        accuracy = accuracies.prod(dim=-1)
        # (batch_size, )
        
        neg_log_lik_classes = [
            -torch.distributions.Categorical( 
            torch.softmax(generative_output[...,i],dim=-1) 
            ).log_prob(target_output[:,i])
            for i in range(n_dim)
        ]
        # (batch_size, n_dim)
        neg_log_lik = torch.stack(neg_log_lik_classes, dim=-1).mean(-1)
        # (batch_size, )
        
        for idx in range(accuracies.shape[-1]):        
            logs_dict[
                f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/generative_referential_game_accuracy/Acc{idx}"] = accuracies[:,idx]

        for idx, nll in enumerate(neg_log_lik_classes):        
            logs_dict[
                f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/generative_referential_game_neg_log_lik/Dim{idx}"] = nll

        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/generative_referential_game_neg_log_lik"] = neg_log_lik
        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/generative_referential_game_accuracy"] = accuracy

        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, loss]
    else:
        raise NotImplementedError
    

class GenerativeListener(Listener):
    def __init__(self,obs_shape, vocab_size=100, max_sentence_length=10, agent_id="l0", logger=None, kwargs=None):
        """
        :param obs_shape: tuple defining the shape of the experience following `(nbr_stimuli, sequence_length, *experience_shape)`
                          where, by default, `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        """
        super(GenerativeListener, self).__init__(agent_id=agent_id, 
                                       obs_shape=obs_shape,
                                       vocab_size=vocab_size,
                                       max_sentence_length=max_sentence_length,
                                       logger=logger, 
                                       kwargs=kwargs)

        self.input_stream_ids["listener"]["experiences"] = "None"
        self.input_stream_ids["listener"]["target_output"] = "None"
            
        self.register_hook(generative_st_gs_referential_game_loss)
        

    def forward(self, sentences, experiences=None, multi_round=False, graphtype="straight_through_gumbel_softmax", tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: None, Tensor of shape `(batch_size, *self.obs_shape)`. 
                        TODO: implement the use of distractors in the reasoning process.
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        if experiences is not None:
            features = self._sense(experiences=experiences, sentences=sentences)
        else:
            features = None 

        if sentences is not None:
            generative_output, listener_temporal_features = self._reason(sentences=sentences, features=features)
        else:
            generative_output = None
            listener_temporal_features = None
        
        next_sentences_widx = None 
        next_sentences_logits = None
        next_sentences = None
        temporal_features = None
        
        if multi_round or ("obverter" in graphtype.lower() and sentences is None):
            utter_outputs = self._utter(features=features, sentences=sentences)
            if len(utter_outputs) == 5:
                next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
            else:
                next_sentences_hidden_states = None
                next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
                        
            if self.training:
                if "gumbel_softmax" in graphtype:    
                    print(f"WARNING: Listener {self.agent_id} is producing messages via a {graphtype}-based graph at the Listener class-level!")
                    if next_sentences_hidden_states is None: 
                        self.tau = self._compute_tau(tau0=tau0)
                        #tau = self.tau.view((-1,1,1)).repeat(1, self.max_sentence_length, self.vocab_size)
                        tau = self.tau.view((-1))
                        # (batch_size)
                    else:
                        self.tau = []
                        for hs in next_sentences_hidden_states:
                            self.tau.append( self._compute_tau(tau0=tau0, h=hs).view((-1)))
                            # list of size batch_size containing Tensors of shape (sentence_length)
                        tau = self.tau 
                        
                    straight_through = (graphtype == "straight_through_gumbel_softmax")

                    next_sentences_stgs = []
                    for bidx in range(len(next_sentences_logits)):
                        nsl_in = next_sentences_logits[bidx]
                        # (sentence_length<=max_sentence_length, vocab_size)
                        tau_in = tau[bidx].view((-1,1))
                        # (1, 1) or (sentence_length, 1)
                        stgs = gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1, eps=self.kwargs["gumbel_softmax_eps"])
                        
                        next_sentences_stgs.append(stgs)
                        #next_sentences_stgs.append( nn.functional.gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                    next_sentences = next_sentences_stgs
                    if isinstance(next_sentences, list): 
                        next_sentences = nn.utils.rnn.pad_sequence(next_sentences, batch_first=True, padding_value=0.0).float()
                        # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)

        output_dict = {"output": generative_output,
                       "generative_output":generative_output, 
                       "sentences_widx":next_sentences_widx, 
                       "sentences_logits":next_sentences_logits, 
                       "sentences_one_hot":next_sentences,
                       #"features":features,
                       "temporal_features": temporal_features
                       }
        
        if not(multi_round):
            self._reset_rnn_states()

        return output_dict 
