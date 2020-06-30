from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np 
import random 

import copy

from .listener import Listener



def havrylov_hinge_learning_signal(decision_logits, target_decision_idx, sampled_decision_idx=None, multi_round=False):
    target_decision_logits = decision_logits.gather(dim=1, index=target_decision_idx)
    # (batch_size, 1)
    
    distractors_logits_list = [torch.cat([pb_dl[:tidx.item()], pb_dl[tidx.item()+1:]], dim=0).unsqueeze(0) 
        for pb_dl, tidx in zip(decision_logits, target_decision_idx)]
    distractors_decision_logits = torch.cat(
        distractors_logits_list, 
        dim=0)
    # (batch_size, nbr_distractors)
    
    loss_element = 1-target_decision_logits+distractors_decision_logits
    # (batch_size, nbr_distractors)
    maxloss_element = torch.max(torch.zeros_like(loss_element), loss_element)
    loss = maxloss_element.sum(dim=1)
    # (batch_size, )

    done = (target_decision_idx == decision_logits.argmax(dim=-1))
    
    return loss, done


def discriminative_st_gs_referential_game_loss(agent,
                                               losses_dict,
                                               input_streams_dict,
                                               outputs_dict,
                                               logs_dict,
                                               **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]
    mode = input_streams_dict["mode"]

    if "listener" not in agent.role:    return outputs_dict

    batch_size = len(input_streams_dict["experiences"])

    sample = input_streams_dict["sample"]
            
    decision_logits = outputs_dict["decision"]
    final_decision_logits = decision_logits
    # (batch_size, max_sentence_length / squeezed if not using obverter agent, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
    if "obverter" in config["graphtype"].lower():
        sentences_lengths = torch.sum(-(input_streams_dict["sentences_widx"].squeeze(-1)-agent.vocab_size).sign(), dim=-1).long()
        # (batch_size,) 
        sentences_lengths = sentences_lengths.reshape(-1,1,1).expand(
            final_decision_logits.shape[0],
            1,
            final_decision_logits.shape[2]
        )
        final_decision_logits = final_decision_logits.gather(dim=1, index=(sentences_lengths-1)).squeeze(1)
    else:
        final_decision_logits = final_decision_logits[:,-1]
    # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
    
    if config["agent_loss_type"].lower() == "nll":
        if config["descriptive"]:  
            decision_probs = F.log_softmax( final_decision_logits, dim=-1)
            criterion = nn.NLLLoss(reduction="none")
            
            if "obverter_least_effort_loss" in config and config["obverter_least_effort_loss"]:
                loss = 0.0
                losses4widx = []
                for widx in range(decision_probs.size(1)):
                    dp = decision_probs[:,widx,...]
                    ls = criterion( dp, sample["target_decision_idx"])
                    loss += config["obverter_least_effort_loss_weights"][widx]*ls 
                    losses4widx.append(ls)
            else:
                decision_probs = decision_probs[:,-1,...]
                loss = criterion( decision_probs, sample["target_decision_idx"])
                # (batch_size, )
        else:   
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            decision_probs = F.log_softmax( final_decision_logits, dim=-1)
            criterion = nn.NLLLoss(reduction="none")
            loss = criterion( decision_probs, sample["target_decision_idx"])
            # (batch_size, )
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, loss]
    
    elif config["agent_loss_type"].lower() == "ce":
        if config["descriptive"]:  
            raise NotImplementedError
        else:   
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            decision_probs = torch.softmax(final_decision_logits, dim=-1)
            criterion = nn.CrossEntropyLoss(reduction="none")
            loss = criterion( final_decision_logits, sample["target_decision_idx"])
            # (batch_size, )
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, loss]
    
    elif config["agent_loss_type"].lower() == "hinge":
        #Havrylov"s Hinge loss:
        # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
        decision_probs = F.log_softmax( final_decision_logits, dim=-1)
        
        loss, _ = havrylov_hinge_learning_signal(decision_logits=final_decision_logits,
                                              target_decision_idx=sample["target_decision_idx"].unsqueeze(1),
                                              multi_round=input_streams_dict["multi_round"])
        # (batch_size, )
        
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, loss]    
    
    outputs_dict["decision_probs"] = decision_probs

    # Accuracy:
    decision_idx = decision_probs.max(dim=-1)[1]
    acc = (decision_idx==sample["target_decision_idx"]).float()*100
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/referential_game_accuracy"] = acc
    outputs_dict["accuracy"] = acc


def penalize_multi_round_binary_reward_fn(sampled_decision_idx, target_decision_idx, decision_logits=None, multi_round=False):
    """
    Computes the reward and done boolean of the current timestep.
    Episode ends if the decisions are correct 
    (or if the max number of round is achieved, but this is handled outside of this function).
    """
    done = (sampled_decision_idx == target_decision_idx)
    r = done.float()*torch.ones_like(target_decision_idx)
    if multi_round:
        r -= 0.1
    return -r, done


class ExperienceBuffer(object):
    def __init__(self, capacity, keys=None, circular_keys={"succ_s":"s"}, circular_offsets={"succ_s":1}):
        """
        Use a different circular offset["succ_s"]=n to implement truncated n-step return...
        """
        if keys is None:    keys = ["s", "a", "r", "non_terminal", "rnn_state"]
        self.keys = keys
        self.circular_keys = circular_keys
        self.circular_offsets = circular_offsets
        self.capacity = capacity
        self.position = dict()
        self.current_size = dict()
        self.reset()

    def add_key(self, key):
        self.keys += [key]
        #setattr(self, key, np.zeros(self.capacity+1, dtype=object))
        setattr(self, key, [None]*(self.capacity+1))
        self.position[key] = 0
        self.current_size[key] = 0

    def add(self, data):
        for k, v in data.items():
            if not(k in self.keys or k in self.circular_keys):  continue
            if k in self.circular_keys: continue
            getattr(self, k)[self.position[k]] = v
            self.position[k] = int((self.position[k]+1) % self.capacity)
            self.current_size[k] = min(self.capacity, self.current_size[k]+1)

    def pop(self):
        """
        Output a data dict of the latest 'complete' data experience.
        """
        all_keys = self.keys+list(self.circular_keys.keys())
        max_offset = 0
        if len(self.circular_offsets):
            max_offset = max([offset for offset in self.circular_offsets.values()])
        data = {k:None for k in self.keys}
        for k in all_keys:
            fetch_k = k
            offset = 0
            if k in self.circular_keys: 
                fetch_k = self.circular_keys[k]
                offset = self.circular_offsets[k]
            next_position_write = self.position[fetch_k] 
            position_complete_read_possible = (next_position_write-1)-max_offset
            k_read_position = position_complete_read_possible+offset 
            data[k] = getattr(self, fetch_k)[k_read_position]
        return data 

    def reset(self):
        for k in self.keys:
            if k in self.circular_keys: continue
            #setattr(self, k, np.zeros(self.capacity+1, dtype=object))
            setattr(self, k, [None]*(self.capacity+1))
            self.position[k] = 0
            self.current_size[k] = 0

    def cat(self, keys, indices=None):
        data = []
        for k in keys:
            assert k in self.keys or k in self.circular_keys, f"Tried to get value from key {k}, but {k} is not registered."
            indices_ = indices
            cidx=0
            if k in self.circular_keys: 
                cidx=self.circular_offsets[k]
                k = self.circular_keys[k]
            v = getattr(self, k)
            if indices_ is None: indices_ = np.arange(self.current_size[k]-1-cidx)
            else:
                # Check that all indices are in range:
                for idx in range(len(indices_)):
                    if self.current_size[k]>0 and indices_[idx]>=self.current_size[k]-1-cidx:
                        indices_[idx] = np.random.randint(self.current_size[k]-1-cidx)
                        # propagate to argument:
                        indices[idx] = indices_[idx]
            """
            """
            indices_ = cidx+indices_
            values = v[indices_]
            data.append(values)
        return data 

    def __len__(self):
        return self.current_size[self.keys[0]]

    def sample(self, batch_size, keys=None):
        if keys is None:    keys = self.keys + self.circular_keys.keys()
        min_current_size = self.capacity
        for idx_key in reversed(range(len(keys))):
            key = keys[idx_key]
            if key in self.circular_keys:   key = self.circular_keys[key]
            if self.current_size[key] == 0:
                continue
            if self.current_size[key] < min_current_size:
                min_current_size = self.current_size[key]

        indices = np.random.choice(np.arange(min_current_size-1), batch_size)
        data = self.cat(keys=keys, indices=indices)
        return data


def compute_reinforce_losses(agent,
                             losses_dict,
                             input_streams_dict,
                             outputs_dict,
                             logs_dict,
                             **kwargs):
    config = kwargs["config"]
    it_rep = kwargs["it_rep"]
    it_comm_round = kwargs["it_comm_round"]

    # then it is the last round, we can compute the loss:
    exp_size = len(agent.exp_buffer)
    R = torch.zeros_like(r)
    returns = []
    for r in reversed(agent.exp_buffer.r[:exp_size]):
        R = r + agent.gamma * R
        returns.insert(0, R.view(-1,1))
        # (batch_size, 1)
    returns = torch.cat(returns, dim=-1)
    # (batch_size, nbr_communication_round)
    # Normalized:
    normalized_learning_signal = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-8)
    # Unnormalized:
    learning_signal = returns #(returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-8)
    # (batch_size, nbr_communication_round)
    
    ls = learning_signal
    if "normalized" in config["graphtype"].lower():
        ls = normalized_learning_signal

    for it_round in range(learning_signal.shape[1]):
        agent.log_dict[f"{agent.agent_id}/learning_signal_{it_round}"] = learning_signal[:,it_round].mean()
    
    formatted_baseline = 0.0
    if "baseline_reduced" in config["graphtype"].lower():
        if agent.training:
            agent.learning_signal_baseline = \
                (agent.learning_signal_baseline*agent.learning_signal_baseline_counter+ls.detach().mean(dim=0)) / \
                (agent.learning_signal_baseline_counter+1)
            agent.learning_signal_baseline_counter += 1
        formatted_baseline = agent.learning_signal_baseline.reshape(1,-1).repeat(batch_size,1).to(learning_signal.device)
    
        for it_round in range(learning_signal.shape[1]):
            agent.log_dict[f"{agent.agent_id}/learning_signal_baseline_{it_round}"] = \
                agent.learning_signal_baseline[it_round].mean()    
    
    # Deterministic:
    listener_decision_loss_deterministic = -learning_signal.sum(dim=-1)
    # (batch_size, )
    listener_decision_loss = listener_decision_loss_deterministic
    
    if "stochastic" in config["graphtype"].lower():
        # Stochastic:
        decision_log_probs = torch.cat(agent.exp_buffer.decision_log_probs[:exp_size], dim=-1)
        # (batch_size, nbr_communication_round)
        listener_decision_loss_stochastic = -(decision_log_probs * (ls.detach()-formatted_baseline)).sum(dim=-1)
        # (batch_size, )
        listener_decision_loss = listener_decision_loss_deterministic+listener_decision_loss_stochastic
        
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss"] = [1.0, listener_decision_loss]
    
    # Decision Entropy loss:
    decision_entropy = torch.cat(agent.exp_buffer.decision_entrs[:exp_size], dim=-1).mean(dim=-1)
    # (batch_size, )
    agent.log_dict[f"{agent.agent_id}/decision_entropy"] = decision_entropy.mean()
    
    speaker_sentences_log_probs = torch.cat(agent.exp_buffer.speaker_sentences_log_probs[:exp_size], dim=-1)
    # (batch_size, max_sentence_length, nbr_communication_round)
    # The log likelihood of each sentences is the sum (in log space) over each next token prediction:
    speaker_sentences_log_probs = speaker_sentences_log_probs.sum(1)
    # (batch_size, nbr_communication_round)
    speaker_policy_loss = -(speaker_sentences_log_probs * (ls.detach()-formatted_baseline)).sum(dim=-1)
    # (batch_size, )
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/speaker_policy_loss"] = [-1.0, speaker_policy_loss]
    
    # Speaker Entropy loss:
    speaker_entropy_loss = -torch.cat(agent.exp_buffer.speaker_sentences_entrs[:exp_size], dim=-1).permute(0,2,1)
    # (batch_size, nbr_communication_round, max_sentence_length)
    speaker_sentences_token_mask = torch.cat(agent.exp_buffer.speaker_sentences_token_mask[:exp_size], dim=-1).permute(0,2,1)
    # (batch_size, nbr_communication_round, max_sentence_length)
    # Sum on the entropy at each token that are not padding: and average over communication round...
    speaker_entropy_loss = (speaker_entropy_loss*speaker_sentences_token_mask).sum(-1).mean(-1)
    # (batch_size, )
    
    speaker_entropy_loss_coeff = 0.0
    if "max_entr" in config["graphtype"]:
        speaker_entropy_loss_coeff = 1e3
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/speaker_entropy_loss"] = [speaker_entropy_loss_coeff, speaker_entropy_loss]
    
    if exp_size > 1:
        #TODO: propagate from speaker to listener!!!

        # Align and reinforce on the listener sentences output:
        # Each listener sentences contribute to the reward of the next round.
        # The last listener sentence does not contribute to anything 
        # (and should not even be computed, as seen by th guard on multi_round above).
        listener_sentences_log_probs = torch.cat(agent.exp_buffer.listener_sentences_log_probs[:exp_size-1], dim=-1)
        # (batch_size, max_sentence_length, nbr_communication_round-1)
        # The log likelihood of each sentences is the sum (in log space) over each next token prediction:
        listener_sentences_log_probs = listener_sentences_log_probs.sum(1)
        # (batch_size, nbr_communication_round-1)
        listener_policy_loss = -(listener_sentences_log_probs * (ls[:,1:].detach()-formatted_baseline[:,1:])).sum(dim=-1)
        # (batch_size, )
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/listener_policy_loss"] = [1.0, listener_policy_loss]    
    
        # Listener Entropy loss:
        listener_entropy_loss = -torch.cat(agent.exp_buffer.listener_sentences_entrs[:exp_size], dim=-1).permute(0,2,1)
        # (batch_size, nbr_communication_round, max_sentence_length)
        listener_sentences_token_mask = torch.cat(agent.exp_buffer.listener_sentences_token_mask[:exp_size], dim=-1).permute(0,2,1)
        # (batch_size, nbr_communication_round, max_sentence_length)
        # Sum on the entropy at each token that are not padding: and average over communication round...
        listener_entropy_loss = (lsitener_entropy_loss*listener_sentences_token_mask).sum(-1).mean(-1)
        # (batch_size, )
        
        listener_entropy_loss_coeff = 0.0
        if "max_entr" in config["graphtype"]:
            listener_entropy_loss_coeff = -1e0
        losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/listener_entropy_loss"] = [listener_entropy_loss_coeff, listener_entropy_loss]    


    
def discriminative_reinforce_referential_game_loss(agent,
                                                   losses_dict,
                                                   input_streams_dict,
                                                   outputs_dict,
                                                   logs_dict,
                                                   **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]
    mode = input_streams_dict["mode"]

    batch_size = len(input_streams_dict["experiences"])

    sample = input_streams_dict["sample"]
            
    decision_logits = outputs_dict["decision"]
    final_decision_logits = decision_logits
    # (batch_size, max_sentence_length / squeezed if not using obverter agent, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
    
    ## ---------------------------------------------------------------------------
    #REINFORCE Policy-gradient algorithm:
    ## ---------------------------------------------------------------------------
    #Compute data:
    final_decision_logits = final_decision_logits[:,-1,...]
    # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
    decision_probs = final_decision_logits.softmax(dim=-1)
    
    decision_distrs = Categorical(probs=decision_probs) 
    decision_entrs = decision_distrs.entropy()
    if "argmax" in config["graphtype"].lower() and not(agent.training):
        sampled_decision_idx = final_decision_logits.argmax(dim=-1).unsqueeze(-1)
    else:
        sampled_decision_idx = decision_distrs.sample().unsqueeze(-1)
    # (batch_size, 1)
    # It is very important to squeeze the sampled indices vector before computing log_prob,
    # otherwise the shape is broadcasted...
    decision_log_probs = decision_distrs.log_prob(sampled_decision_idx.squeeze()).unsqueeze(-1)
    # (batch_size, 1)
    
    # Learning signal:
    target_decision_idx = sample["target_decision_idx"].unsqueeze(1)
    # (batch_size, 1)
    learning_signal, done = agent.learning_signal_pred_fn(sampled_decision_idx=sampled_decision_idx,
                                           decision_logits=final_decision_logits,
                                           target_decision_idx=target_decision_idx,
                                           multi_round=input_streams_dict["multi_round"])
    # Frame the learning loss as penalty:
    r = -learning_signal

    # Where are the actual token (different from padding):
    speaker_sentences_token_mask = (input_streams_dict["sentences_widx"] != agent.vocab_pad_idx)
    # (batch_size, max_sentence_length, 1)
    
    ## ---------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------
    
    # Compute sentences log_probs:
    speaker_sentences_logits = input_streams_dict["sentences_logits"]
    # (batch_size, max_sentence_length, vocab_size)
    pad_token_logit = torch.zeros_like(speaker_sentences_logits[0][0]).unsqueeze(0)
    pad_token_logit[:, agent.vocab_pad_idx] = 1.0
    # (1, vocab_size,)

    for b in range(len(speaker_sentences_logits)):
        remainder = agent.kwargs["max_sentence_length"] - len(speaker_sentences_logits[b])
        if remainder > 0:
            speaker_sentences_logits[b] = torch.cat([speaker_sentences_logits[b], pad_token_logit.repeat(remainder,1)], dim=0)
        speaker_sentences_logits[b] = speaker_sentences_logits[b].unsqueeze(0)
    speaker_sentences_logits = torch.cat(speaker_sentences_logits, dim=0)

    speaker_sentences_log_probs = F.log_softmax(speaker_sentences_logits, dim=-1)
    # (batch_size, max_sentence_length, vocab_size)
    speaker_sentences_probs = speaker_sentences_log_probs.softmax(dim=-1)
    # (batch_size, max_sentence_length, vocab_size)
    speaker_sentences_log_probs = speaker_sentences_log_probs.gather(dim=-1, 
        index=input_streams_dict["sentences_widx"].long())
    # (batch_size, max_sentence_length, 1)
    speaker_sentences_probs = speaker_sentences_probs.gather(dim=-1, 
        index=input_streams_dict["sentences_widx"].long())
    # (batch_size, max_sentence_length, 1)
    
    # Entropy:
    speaker_sentences_entrs = -(speaker_sentences_probs * speaker_sentences_log_probs)
    # (batch_size, max_sentence_length, 1)
    
    ## ---------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------
    
    listener_sentences_log_probs = None
    listener_sentences_entrs = None
    listener_sentences_token_mask = None 

    if input_streams_dict["multi_round"]:  
        # Where are the actual token (different from padding):
        listener_sentences_token_mask = (outputs_dict["sentences_widx"] != agent.vocab_pad_idx)
        # (batch_size, max_sentence_length, 1)
        
        listener_sentences_logits = outputs_dict["sentences_logits"]
        # (batch_size, max_sentence_length, vocab_size)
        for b in range(len(listener_sentences_logits)):
            remainder = agent.kwargs["max_sentence_length"] - len(listener_sentences_logits[b])
            if remainder > 0:
                listener_sentences_logits[b] = torch.cat([listener_sentences_logits[b], pad_token_logit.repeat(remainder,1)], dim=0)
            listener_sentences_logits[b] = listener_sentences_logits[b].unsqueeze(0)
        listener_sentences_logits = torch.cat(listener_sentences_logits, dim=0)
        listener_sentences_log_probs = F.log_softmax(listener_sentences_logits, dim=-1)
        # (batch_size, max_sentence_length, vocab_size)
        listener_sentences_probs = listener_sentences_log_probs.softmax(dim=-11)
        # (batch_size, max_sentence_length, vocab_size)
        listener_sentences_log_probs = listener_sentences_log_probs.gather(dim=-1, 
            index=outputs_dict["sentences_widx"].long())
        # (batch_size, max_sentence_length, 1)
        listener_sentences_probs = listener_sentences_probs.gather(dim=-1, 
            index=input_streams_dict["sentences_widx"].long())
        # (batch_size, max_sentence_length, 1)
        
        # Entropy:
        listener_sentences_entrs = -(listener_sentences_probs * listener_sentences_log_probs)
        # (batch_size, max_sentence_length, 1)
                
    ## ---------------------------------------------------------------------------
    # Record data:
    ## ---------------------------------------------------------------------------
    
    data = {"speaker_sentences_entrs":speaker_sentences_entrs,
            "speaker_sentences_token_mask":speaker_sentences_token_mask,
            "speaker_sentences_log_probs":speaker_sentences_log_probs,
            "listener_sentences_entrs":listener_sentences_entrs,
            "listener_sentences_token_mask":listener_sentences_token_mask,
            "listener_sentences_log_probs":listener_sentences_log_probs,
            "decision_entrs":decision_entrs,
            "decision_log_probs":decision_log_probs,
            "r":r,
            "done":done}

    agent.exp_buffer.add(data)

    ## ---------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------
    
    # Compute losses:
    if not(input_streams_dict["multi_round"]):
        compute_reinforce_losses(
            agent=agent,
            losses_dict=losses_dict,
            input_streams_dict=input_streams_dict,
            outputs_dict=outputs_dict,
            logs_dict=logs_dict,
            config=config,
            it_rep=it_rep,
            it_comm_round=it_comm_round
        )
        agent.exp_buffer.reset()
                    
    outputs_dict["decision_probs"] = decision_probs

    # Accuracy:
    decision_idx = decision_probs.max(dim=-1)[1]
    acc = (decision_idx==sample["target_decision_idx"]).float()*100
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/referential_game_accuracy"] = acc
    outputs_dict["accuracy"] = acc



class DiscriminativeListener(Listener):
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
        super(DiscriminativeListener, self).__init__(agent_id=agent_id, 
                                       obs_shape=obs_shape,
                                       vocab_size=vocab_size,
                                       max_sentence_length=max_sentence_length,
                                       logger=logger, 
                                       kwargs=kwargs)

        if "reinforce" in self.kwargs["graphtype"]:
            # REINFORCE algorithm:
            self.gamma = 0.99
            #self.learning_signal_pred_fn = penalize_multi_round_binary_loss_fn
            self.learning_signal_pred_fn = havrylov_hinge_learning_signal
            self.exp_buffer = ExperienceBuffer(capacity=self.kwargs["nbr_communication_round"]*2,
                                               keys=["speaker_sentences_entrs",
                                                     "speaker_sentences_token_mask",
                                                     "speaker_sentences_log_probs",
                                                     "listener_sentences_entrs",
                                                     "listener_sentences_log_probs",
                                                     "decision_entrs",
                                                     "decision_log_probs",
                                                     "r",
                                                     "done"],
                                               circular_keys={}, 
                                               circular_offsets={})

            self.learning_signal_baseline = 0.0
            self.learning_signal_baseline_counter = 0

            self.register_hook(discriminative_reinforce_referential_game_loss)
        else:
            self.register_hook(discriminative_st_gs_referential_game_loss)
        

    def _compute_tau(self, tau0):
        raise NotImplementedError
        
    def _sense(self, experiences, sentences=None):
        """
        Infers features from the experiences that have been provided.

        :param exp: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        """
        raise NotImplementedError

    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        raise NotImplementedError
    
    def _utter(self, features, sentences):
        """
        Reasons about the features and the listened sentences to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
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
        batch_size = experiences.size(0)
        features = self._sense(experiences=experiences, sentences=sentences)
        if sentences is not None:
            decision_logits, listener_temporal_features = self._reason(sentences=sentences, features=features)
        else:
            decision_logits = None
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

        output_dict = {"output": decision_logits,
                       "decision": decision_logits, 
                       "sentences_widx":next_sentences_widx, 
                       "sentences_logits":next_sentences_logits, 
                       "sentences_one_hot":next_sentences,
                       #"features":features,
                       "temporal_features": temporal_features
                       }
        
        if not(multi_round):
            self._reset_rnn_states()

        return output_dict 