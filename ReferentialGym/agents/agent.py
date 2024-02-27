from typing import Dict, List, Optional 

import os 
import random 
import copy
from ordered_set import OrderedSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision

import numpy as np 

from ..modules import Module

import wandb


def vae_loss_hook(agent,
                  losses_dict,
                  input_streams_dict,
                  outputs_dict,
                  logs_dict,
                  **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    
    if isinstance(agent.VAE_losses, torch.Tensor):
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
    

wandb_logging_table = None
def wandb_ImageOrGIF(data):
    if data.shape[0] == 3:
        return wandb.Image(data.transpose(1,2))
    nbr_frames = data.shape[0] // 3
    data = data.reshape(3, nbr_frames, *data.shape[-2:]).transpose(0,1)
    #.transpose(2,3)
    if data.max().item() <= 1.0:
        data = data*255
    return wandb.Video(data, fps=1, format='gif')

def wandb_logging_hook(
    agent,
    losses_dict,
    input_streams_dict,
    outputs_dict,
    logs_dict,
    **kwargs,
):
    if 'listener' not in agent.role:  return
    
    global_it = input_streams_dict['global_it_comm_round']
    mode = input_streams_dict['mode']

    if global_it % 16384 != 0:    return
    
    idx2w = agent.idx2w

    listener_experiences = input_streams_dict['experiences']
    listener_experience_indices = input_streams_dict['sample']['listener_indices']
    batch_size = listener_experience_indices.shape[0]
    listener_experience_indices = listener_experience_indices.reshape((batch_size, -1))

    speaker_experiences = input_streams_dict['sample']['speaker_experiences']
    #listener_experiences = input_streams_dict['sample']['listener_experiences']
    nbr_distractors_po = listener_experiences.shape[1]
    
    #imgs = experiences.reshape((-1, *experiences.shape[-3:]))
    #grid_imgs = torchvision.utils.make_grid(imgs, nrow=nbr_distractors_po)
    
    sentences = input_streams_dict['sentences_widx']
    max_sentence_length = agent.config['max_sentence_length']
    
    target_indices = input_streams_dict['sample']['target_decision_idx']

    global wandb_logging_table
    if wandb_logging_table is None:
        columns = [
            "it",
            "mode",
            "bidx",
            "sentence",
            "target_stimulus",
        ]
        for didx in range(nbr_distractors_po):
            columns.append(f"distr_stimulus_{didx}")
        for didx in range(nbr_distractors_po):
            columns.append(f"idx_distr_stimulus_{didx}")
        for tidx in range(max_sentence_length):
            columns.append(f"token_{tidx}")

        wandb_logging_table = wandb.Table(columns)
    
    for bidx in range(listener_experiences.shape[0]):
        data = []
        data.append(global_it)
        data.append(mode)
        data.append(bidx)
        sentence = sentences[bidx].cpu().reshape(max_sentence_length).numpy().tolist()
        if idx2w is not None:
            sentence = [idx2w[t] for t in sentence]
            data.append(' '.join(sentence))
        else:
            data.append(sentence)
        try:
            target_stimulus = speaker_experiences[bidx,0,0].cpu().transpose(1,2)#*255
            target_stimulus = wandb_ImageOrGIF(target_stimulus)
        except Exception as e:
            print(e)
            target_stimulus = speaker_experiences[bidx].cpu().numpy().tolist()
        data.append(target_stimulus)
        for didx in range(nbr_distractors_po):
            try:
                dstimulus = listener_experiences[bidx,didx,0].cpu()#*255
                dstimulus = wandb_ImageOrGIF(dstimulus.transpose(1,2))
            except Exception as e:
                print(e)
                dstimulus = listener_experiences[bidx,didx].cpu().numpy().tolist()
            data.append(dstimulus)
        for didx in range(nbr_distractors_po):
            dstim_idx = listener_experience_indices[bidx,didx].cpu().item()
            data.append(dstim_idx)
        for tidx in range(max_sentence_length):
            token = sentence[tidx]
            data.append(token)
        wandb_logging_table.add_data(*data)
    
    wandb.log({f"{mode}/WandbLoggingTable":wandb_logging_table}, commit=False)
    wandb_logging_table = None
    
    return 

    
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
            "indices":"current_dataloader:sample:speaker_indices", 
            "exp_latents":"current_dataloader:sample:speaker_exp_latents", 
            "exp_latents_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
            "exp_latents_values":"current_dataloader:sample:speaker_exp_latents_values", 
            "sentences_logits":"modules:current_listener:sentences_logits",
            "sentences_one_hot":"modules:current_listener:sentences_one_hot",
            "sentences_widx":"modules:current_listener:sentences_widx", 
            "config":"config",
            "graphtype":"config:graphtype",
            "tau0":"config:tau0",
            "running_train_accuracy":"signals:running_train_accuracy",
            "running_test_accuracy":"signals:running_test_accuracy",
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
            "indices":"current_dataloader:sample:listener_indices", 
            "exp_latents":"current_dataloader:sample:listener_exp_latents", 
            "exp_latents_one_hot_encoded":"current_dataloader:sample:listener_exp_latents_one_hot_encoded", 
            "exp_latents_values":"current_dataloader:sample:listener_exp_latents_values", 
            "sentences_logits":"modules:current_speaker:sentences_logits",
            "sentences_one_hot":"modules:current_speaker:sentences_one_hot",
            "sentences_widx":"modules:current_speaker:sentences_widx", 
            "config":"config",
            "graphtype":"config:graphtype",
            "tau0":"config:tau0",
            "running_train_accuracy":"signals:running_train_accuracy",
            "running_test_accuracy":"signals:running_test_accuracy",
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
        
        vocabulary = OrderedSet('key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'.split(' ')
        )
        self.vocabulary = OrderedSet([w.lower() for w in vocabulary])
        self.vocabulary = list(self.vocabulary)
        while len(self.vocabulary) < self.vocab_size+2:
            self.vocabulary.append( f"DUMMY{len(self.vocabulary)}")
        self.vocabulary = self.vocabulary[:self.vocab_size+2]
        self.vocabulary = list(OrderedSet(self.vocabulary))
        self.vocabulary = ['EoS', 'SoS'] + self.vocabulary
        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w 
        
        print(type(self))
        print(self.idx2w)

        self.max_sentence_length = max_sentence_length
        
        self.logger = logger 
        self.kwargs = kwargs

        self.log_idx = 0
        self.log_dict = dict()

        self.use_sentences_one_hot_vectors = False

        self.vocab_stop_idx = 0
        self.vocab_pad_idx = self.vocab_size
        
        self.hooks = []
        if self.kwargs.get("with_weight_maxl1_loss", False):
            self.register_hook(maxl1_loss_hook)
        self.register_hook(wandb_logging_hook)
        
        self.role = role        
    
    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w 
        
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
    
    def _log(self, log_dict, batch_size, it_rep=0):
        if self.logger is None: 
            return 

        entry_id = f"{self.agent_id}/rep{it_rep}"
        agent_log_dict = {entry_id: dict()}
        for key, data in log_dict.items():
            if data is None:
                data = [None]*batch_size
            agent_log_dict[entry_id].update({f"{key}":data})
        
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

    def forward(self, sentences, experiences, multi_round=False, graphtype="straight_through_gumbel_softmax", tau0=0.2, **kwargs):
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
        config = input_streams_dict.get("config", None)
        mode = input_streams_dict.get("mode", 'train')
        it_rep = input_streams_dict.get("it_rep", 0)
        it_comm_round = input_streams_dict.get("it_comm_round", 0)
        global_it_comm_round = input_streams_dict.get("global_it_comm_round", 0)
        
        losses_dict = input_streams_dict.get("losses_dict", {})
        logs_dict = input_streams_dict.get("logs_dict", {})
        
        self.sample = input_streams_dict.get("sample", {})
        self.experiences = input_streams_dict["experiences"]
        if isinstance(self.experiences, list):  
            self.experiences = self.experiences[0]
            while len(self.experiences.shape) < 4:
                self.experiences = self.experiences.unsqueeze(1)
        
        self.indices = input_streams_dict.get("indices", None)
        self.exp_latents = input_streams_dict.get("exp_latents", None)
        self.exp_latents_values = input_streams_dict.get("exp_latents_values", None)


        input_sentence = input_streams_dict.get("sentences_widx", None)
        if self.use_sentences_one_hot_vectors:
            input_sentence = input_streams_dict.get("sentences_one_hot", None)
        
        if isinstance(input_sentence, list):  
            input_sentence = input_sentence[0]
        
        assert self.experiences is not None or input_sentence is not none
        if self.experiences is not None:
            batch_size = self.experiences.shape[0]
        else:
            batch_size = input_sentence.shape[0]
            
        outputs_dict = self(
            sentences=input_sentence,
            experiences=self.experiences,
            multi_round=input_streams_dict.get("multi_round", False),
            graphtype=input_streams_dict.get("graphtype", self.kwargs['graphtype']),
            tau0=input_streams_dict.get("tau0", self.kwargs['tau0']),
            sample=input_streams_dict.get("sample", None),
        )

        for key in self.output_stream_ids:
            if key in outputs_dict:
                output_key = self.output_stream_ids[key]
                outputs_dict[output_key] = [outputs_dict[key]]

        if self.exp_latents is not None:
            outputs_dict["exp_latents"] = input_streams_dict["exp_latents"]
            outputs_dict["exp_latents_values"] = input_streams_dict["exp_latents_values"]
            outputs_dict["exp_latents_one_hot_encoded"] = input_streams_dict["exp_latents_one_hot_encoded"]
        self._log(outputs_dict, batch_size=batch_size, it_rep=it_rep)

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


