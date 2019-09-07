import torch
import torch.nn as nn

import copy 

from ..networks import layer_init
from ..utils import gumbel_softmax 
from .agent import Agent

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
        super(Listener, self).__init__(agent_id=agent_id, logger=logger, kwargs=kwargs)
        self.obs_shape = obs_shape
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        
        # Multi-round:
        self._reset_rnn_states()

    def reset(self):
        self.apply(layer_init)

    def _reset_rnn_states(self):
        self.rnn_states = None 

    def _tidyup(self):
        pass 

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
        features = self._sense(experiences=experiences, sentences=sentences)
        if sentences is not None:
            decision_logits, temporal_features = self._reason(sentences=sentences, features=features)
        else:
            decision_logits = None
            temporal_features = None 
        
        next_sentences_widx = None 
        next_sentences_logits = None
        next_sentences = None

        if multi_round or ('obverter' in graphtype.lower() and sentences is None):
            next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = self._utter(features=features, sentences=sentences)
            
            if self.training:
                if 'gumbel_softmax' in graphtype:
                    self.tau = self._compute_tau(tau0=tau0)
                    #tau = self.tau.view((-1,1,1)).repeat(1,1,self.vocab_size)
                    tau = self.tau.view((-1))

                    straight_through = ('straight_through_gumbel_softmax' in graphtype)
                    
                    next_sentences_stgs = []
                    for bidx in range(len(next_sentences_logits)):
                        nsl_in = next_sentences_logits[bidx]
                        tau_in = tau[bidx]
                        next_sentences_stgs.append( gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                        #next_sentences_stgs.append( nn.functional.gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                    next_sentences = next_sentences_stgs
                    if isinstance(next_sentences, list): 
                        next_sentences = nn.utils.rnn.pad_sequence(next_sentences, batch_first=True, padding_value=0.0).float()
                        # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)
        
        output_dict = {'decision': decision_logits, 
                       'sentences_widx':next_sentences_widx, 
                       'sentences_logits':next_sentences_logits, 
                       'sentences_one_hot':next_sentences,
                       #'features':features,
                       'temporal_features': temporal_features
                       }
        
        if not(multi_round):
            self._reset_rnn_states()

        self._tidyup()
        self._log(output_dict)

        return output_dict 
