from typing import Dict, List 

import os 
import random 
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np 

from ..modules import Module


def vae_loss_hook(agent,
                  losses_dict,
                  input_streams_dict,
                  outputs_dict,
                  logs_dict,
                  **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/{agent.role}/VAE_loss"] = [agent.kwargs["VAE_lambda"], agent.VAE_losses]

def maxl1_loss_hook(agent,
                    losses_dict,
                    input_streams_dict,
                    outputs_dict,
                    logs_dict,
                    **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    
    weight_maxl1_loss = 0.0
    for p in agent.parameters() :
        weight_maxl1_loss += torch.max( torch.abs(p) )
    outputs_dict["maxl1_loss"] = weight_maxl1_loss        

    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/{agent.role}_maxl1_weight_loss"] = [1.0, weight_maxl1_loss]
    



class Agent(Module):
    def __init__(self, 
                 agent_id="l0", 
                 obs_shape=[1,1,1,32,32], 
                 vocab_size=100, 
                 max_sentence_length=10, 
                 logger=None, 
                 kwargs=None,
                 role=None):
        """
        :param agent_id: str defining the ID of the agent over the population.
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        :param role: str defining the role of the agent, e.g. "speaker"/"listener".
        """

        input_stream_ids = {"speaker":list(), "listener":list()}
        
        input_stream_ids["speaker"] = {
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "exp_latents":"current_dataloader:sample:speaker_exp_latents", 
            "exp_latents_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
            "exp_latents_values":"current_dataloader:sample:speaker_exp_latents_values", 
            "sentences_logits":"modules:current_listener:sentences_logits",
            "sentences_one_hot":"modules:current_listener:sentences_one_hot",
            "sentences_widx":"modules:current_listener:sentences_widx", 
            "config":"config",
            "graphtype":"config:graphtype",
            "tau0":"config:tau0",
            "multi_round":"signals:multi_round",
            "end_of_epoch_sample":"signals:end_of_epoch_sample",
            "mode":"signals:mode",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "global_it_comm_round":"signals:global_it_step",
            "sample":"current_dataloader:sample",
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
        }

        input_stream_ids["listener"] = {
            "experiences":"current_dataloader:sample:listener_experiences", 
            "exp_latents":"current_dataloader:sample:listener_exp_latents", 
            "exp_latents_one_hot_encoded":"current_dataloader:sample:listener_exp_latents_one_hot_encoded", 
            "exp_latents_values":"current_dataloader:sample:listener_exp_latents_values", 
            "sentences_logits":"modules:current_speaker:sentences_logits",
            "sentences_one_hot":"modules:current_speaker:sentences_one_hot",
            "sentences_widx":"modules:current_speaker:sentences_widx", 
            "config":"config",
            "graphtype":"config:graphtype",
            "tau0":"config:tau0",
            "multi_round":"signals:multi_round",
            "end_of_epoch_sample":"signals:end_of_epoch_sample",
            "mode":"signals:mode",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "global_it_comm_round":"signals:global_it_step",
            "sample":"current_dataloader:sample",
            "losses_dict":"losses_dict",
            "logs_dict":"logs_dict",
        }

        super(Agent, self).__init__(id=agent_id,
                                    type="Agent", 
                                    config=kwargs,
                                    input_stream_ids=input_stream_ids)
        
        self.agent_id = agent_id
        
        self.obs_shape = obs_shape
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        
        self.logger = logger 
        self.kwargs = kwargs

        self.log_idx = 0
        self.log_dict = dict()

        self.use_sentences_one_hot_vectors = False

        self.vocab_stop_idx = 0
        self.vocab_pad_idx = self.vocab_size
        
        self.hooks = []
        if self.kwargs["with_weight_maxl1_loss"]:
            self.register_hook(maxl1_loss_hook)

        self.role = role        
    
    def get_input_stream_ids(self):
        return self.input_stream_ids[self.role]

    def clone(self, clone_id="a0"):
        logger = self.logger
        self.logger = None 
        clone = copy.deepcopy(self)
        clone.agent_id = clone_id
        clone.logger = logger 
        self.logger = logger  
        return clone 

    def save(self, path, filename=None):
        logger = self.logger
        self.logger = None
        if filename is None:
            filepath = path+self.id+".agent"
        else:
            filepath = os.path.join(path, filename)
        torch.save(self, filepath)
        self.logger = logger 

    def _tidyup(self):
        pass 
    
    def _log(self, log_dict, batch_size):
        if self.logger is None: 
            return 

        agent_log_dict = {f"{self.agent_id}": dict()}
        for key, data in log_dict.items():
            if data is None:
                data = [None]*batch_size
            agent_log_dict[f"{self.agent_id}"].update({f"{key}":data})
        
        self.logger.add_dict(agent_log_dict, batch=True, idx=self.log_idx) 
        
        self.log_idx += 1

    def register_hook(self, hook):
        self.hooks.append(hook)

    def embed_sentences(self, sentence):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :returns embedded_sentences: Tensor of shape `(batch_size, max_sentence_length, symbol_embedding_size)` containing the padded sequence of embedded symbols.
        """
        raise NotImplementedError

    def forward(self, sentences, experiences, multi_round=False, graphtype="straight_through_gumbel_softmax", tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        raise NotImplementedError

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Compute the losses and return them along with the produced outputs.

        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_logits'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits over symbols.
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sentences_one_hot'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
            - `'exp_latents'`: Tensor of shape `(batch_size, nbr_latent_dimensions)`.
            - `'multi_round'`: Boolean defining whether to utter a sentence back or not.
            - `'graphtype'`: String defining the type of symbols used in the output sentence:
                        - `'categorical'`: one-hot-encoded symbols.
                        - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                        - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                        - `'obverter'`: obverter training scheme...
            - `'tau0'`: Float, temperature with which to apply gumbel-softmax estimator. 
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'config'`: Dict of hyperparameters to the referential game.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
            - `'it'`: Integer specifying the iteration number of the current function call.
        """
        config = input_streams_dict["config"]
        mode = input_streams_dict["mode"]
        it_rep = input_streams_dict["it_rep"]
        it_comm_round = input_streams_dict["it_comm_round"]
        global_it_comm_round = input_streams_dict["global_it_comm_round"]
        
        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]
        
        input_sentence = input_streams_dict["sentences_widx"]
        if self.use_sentences_one_hot_vectors:
            input_sentence = input_streams_dict["sentences_one_hot"]

        if input_streams_dict["experiences"] is not None:
            batch_size = input_streams_dict["experiences"].shape[0]
        else:
            batch_size = input_sentence.shape[0]
            
        outputs_dict = self(sentences=input_sentence,
                           experiences=input_streams_dict["experiences"],
                           multi_round=input_streams_dict["multi_round"],
                           graphtype=input_streams_dict["graphtype"],
                           tau0=input_streams_dict["tau0"])

        outputs_dict["exp_latents"] = input_streams_dict["exp_latents"]
        outputs_dict["exp_latents_values"] = input_streams_dict["exp_latents_values"]
        outputs_dict["exp_latents_one_hot_encoded"] = input_streams_dict["exp_latents_one_hot_encoded"]
        self._log(outputs_dict, batch_size=batch_size)

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//

        """
        if hasattr(self, "TC_losses"):
            losses_dict[f"{self.role}/TC_loss"] = [1.0, self.TC_losses]
        """
        if hasattr(self, "VAE_losses") and vae_loss_hook not in self.hooks:
            self.register_hook(vae_loss_hook)

        if hasattr(self,"tau"): 
            tau = torch.cat([ t.view((-1)) for t in self.tau], dim=0) if isinstance(self.tau, list) else self.tau
            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/Tau/{self.agent_id}"] = tau
        
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        
        for hook in self.hooks:
            hook(
                agent=self,
                losses_dict=losses_dict,
                input_streams_dict=input_streams_dict,
                outputs_dict=outputs_dict,
                logs_dict=logs_dict
            )

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        
        # Logging:        
        for logname, value in self.log_dict.items():
            self.logger.add_scalar(f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.role}/{logname}", value.item(), global_it_comm_round)
        self.log_dict = {}

        self._tidyup()
        
        outputs_dict["losses"] = losses_dict

        return outputs_dict    