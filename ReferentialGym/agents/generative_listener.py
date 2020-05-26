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


def generative_st_gs_referential_game_loss(agent,
                                           losses_dict,
                                           input_streams_dict,
                                           outputs_dict,
                                           logs_dict,
                                           **kwargs):
    it_rep = input_streams_dict['it_rep']
    it_comm_round = input_streams_dict['it_comm_round']
    config = input_streams_dict['config']
    mode = input_streams_dict['mode']

    batch_size = len(input_streams_dict['experiences'])

    sample = input_streams_dict['sample']
            
    generative_output = outputs_dict['generative_output']
    final_decision_logits = decision_logits
    # (batch_size, max_sentence_length / squeezed if not using obverter agent, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
    if 'obverter' in config['graphtype'].lower():
        sentences_lengths = torch.sum(-(input_streams_dict['sentences_widx'].squeeze(-1)-agent.vocab_size).sign(), dim=-1).long()
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
    
    if config['agent_loss_type'].lower() == 'nll':
        if config['descriptive']:  
            decision_probs = F.log_softmax( final_decision_logits, dim=-1)
            criterion = nn.NLLLoss(reduction='none')
            
            if 'obverter_least_effort_loss' in config and config['obverter_least_effort_loss']:
                loss = 0.0
                losses4widx = []
                for widx in range(decision_probs.size(1)):
                    dp = decision_probs[:,widx,...]
                    ls = criterion( dp, sample['target_decision_idx'])
                    loss += config['obverter_least_effort_loss_weights'][widx]*ls 
                    losses4widx.append(ls)
            else:
                decision_probs = decision_probs[:,-1,...]
                loss = criterion( decision_probs, sample['target_decision_idx'])
                # (batch_size, )
        else:   
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            decision_probs = F.log_softmax( final_decision_logits, dim=-1)
            criterion = nn.NLLLoss(reduction='none')
            loss = criterion( decision_probs, sample['target_decision_idx'])
            # (batch_size, )
        losses_dict[f'repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss'] = [1.0, loss]
    
    elif config['agent_loss_type'].lower() == 'ce':
        if config['descriptive']:  
            raise NotImplementedError
        else:   
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            decision_probs = torch.softmax(final_decision_logits, dim=-1)
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion( final_decision_logits, sample['target_decision_idx'])
            # (batch_size, )
        losses_dict[f'repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss'] = [1.0, loss]
    
    elif config['agent_loss_type'].lower() == 'hinge':
        #Havrylov's Hinge loss:
        # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
        decision_probs = F.log_softmax( final_decision_logits, dim=-1)
        
        loss, _ = havrylov_hinge_learning_signal(decision_logits=final_decision_logits,
                                              target_decision_idx=sample['target_decision_idx'].unsqueeze(1),
                                              multi_round=input_streams_dict['multi_round'])
        # (batch_size, )
        
        losses_dict[f'repetition{it_rep}/comm_round{it_comm_round}/referential_game_loss'] = [1.0, loss]    
    
    outputs_dict['decision_probs'] = decision_probs

    # Accuracy:
    decision_idx = decision_probs.max(dim=-1)[1]
    acc = (decision_idx==sample['target_decision_idx']).float()*100
    logs_dict[f'{mode}/repetition{it_rep}/comm_round{it_comm_round}/referential_game_accuracy'] = acc
    outputs_dict['accuracy'] = acc


class GenerativeListener(Listener):
    def __init__(self,obs_shape, vocab_size=100, max_sentence_length=10, agent_id='l0', logger=None, kwargs=None):
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

        self.register_hook(generative_st_gs_referential_game_loss)
        

    def forward(self, sentences, experiences, multi_round=False, graphtype='straight_through_gumbel_softmax', tau0=0.2):
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
            generative_output, listener_temporal_features = self._reason(sentences=sentences, features=features)
        else:
            generative_output = None
            listener_temporal_features = None
        
        next_sentences_widx = None 
        next_sentences_logits = None
        next_sentences = None
        temporal_features = None
        
        if multi_round or ('obverter' in graphtype.lower() and sentences is None):
            utter_outputs = self._utter(features=features, sentences=sentences)
            if len(utter_outputs) == 5:
                next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
            else:
                next_sentences_hidden_states = None
                next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
                        
            if self.training:
                if 'gumbel_softmax' in graphtype:    
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
                        
                    straight_through = (graphtype == 'straight_through_gumbel_softmax')

                    next_sentences_stgs = []
                    for bidx in range(len(next_sentences_logits)):
                        nsl_in = next_sentences_logits[bidx]
                        # (sentence_length<=max_sentence_length, vocab_size)
                        tau_in = tau[bidx].view((-1,1))
                        # (1, 1) or (sentence_length, 1)
                        stgs = gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1, eps=self.kwargs['gumbel_softmax_eps'])
                        
                        next_sentences_stgs.append(stgs)
                        #next_sentences_stgs.append( nn.functional.gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                    next_sentences = next_sentences_stgs
                    if isinstance(next_sentences, list): 
                        next_sentences = nn.utils.rnn.pad_sequence(next_sentences, batch_first=True, padding_value=0.0).float()
                        # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)

        output_dict = {'output': generative_output,
                       'generative_output':generative_output, 
                       'sentences_widx':next_sentences_widx, 
                       'sentences_logits':next_sentences_logits, 
                       'sentences_one_hot':next_sentences,
                       #'features':features,
                       'temporal_features': temporal_features
                       }
        
        if not(multi_round):
            self._reset_rnn_states()

        return output_dict 
