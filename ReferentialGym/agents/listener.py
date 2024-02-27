import torch
import torch.nn as nn

import copy 

from ..networks import layer_init
from ..utils import gumbel_softmax 
from .agent import Agent

#TODO:
"""
if 'iterated_learning_scheme' in config \
    and config['iterated_learning_scheme']\
    and 'iterated_learning_rehearse_MDL' in config \
    and config['iterated_learning_rehearse_MDL']:
    # Rehearsing:
    listener_speaking_outputs = agent(experiences=sample['speaker_experiences'], 
                                     sentences=None, 
                                     multi_round=input_streams_dict['multi_round'],
                                     graphtype=input_streams_dict['graphtype'],
                                     tau0=input_streams_dict['tau0'])
    # Let us enforce the Minimum Description Length Principle:
    # Listener's speaking entropy:
    listener_sentences_log_probs = [s_logits.reshape(-1,agent.vocab_size).log_softmax(dim=-1) for s_logits in listener_speaking_outputs['sentences_logits']]
    listener_sentences_log_probs = torch.cat(
        [s_log_probs.gather(dim=-1,index=s_widx[:s_log_probs.shape[0]].long()).sum().unsqueeze(0) 
        for s_log_probs, s_widx in zip(listener_sentences_log_probs, listener_speaking_outputs['sentences_widx'])], 
        dim=0)
    listener_entropies_per_sentence = -(listener_sentences_log_probs.exp() * listener_sentences_log_probs)
    # (batch_size, )
    # Maximization:
    losses_dict[f'repetition{it_rep}/comm_round{it_comm_round}/ilm_MDL_loss'] = [-config['iterated_learning_rehearse_MDL_factor'], listener_entropies_per_sentence]

    '''
    listener_speaking_entropies = [torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy().mean().view(-1) for w_logits in s_logits], dim=0) for s_logits in listener_speaking_outputs['sentences_logits']]
    # List of size batch_size of Tensor of shape (sentence_length,)
    per_sentence_max_entropies = torch.stack([ lss.max(dim=0)[0] for lss in listener_speaking_entropies])
    # Tensor of shape (batch_size,1)
    ilm_loss = per_sentence_max_entropies.mean(dim=-1)
    # (batch_size, )
    losses_dict['ilm_MDL_loss'] = [1.0, ilm_loss]
    '''

if 'with_listener_entropy_regularization' in config and config['with_listener_entropy_regularization']:
    entropies = torch.cat([ torch.distributions.categorical.Categorical(logits=d_logits).entropy().view(1) for d_logits in final_decision_logits], dim=-1)
    losses_dict[f'repetition{it_rep}/comm_round{it_comm_round}/listener_entropy_loss'] = [config['entropy_regularization_factor'], entropies_per_decision.squeeze()]
    # (batch_size, )
"""



class Listener(Agent):
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
        super(Listener, self).__init__(agent_id=agent_id, 
                                       obs_shape=obs_shape,
                                       vocab_size=vocab_size,
                                       max_sentence_length=max_sentence_length,
                                       logger=logger, 
                                       kwargs=kwargs,
                                       role="listener")
        
        # Multi-round:
        self._reset_rnn_states()

    def reset_weights(self):
        self.apply(layer_init)

    def _reset_rnn_states(self):
        self.rnn_states = None 

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
    
    def _utter(self, features, sentences, rnn_states):
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

    
