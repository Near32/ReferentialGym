import torch
import torch.nn as nn

from ..networks import layer_init


class Listener(nn.Module):
    def __init__(self,obs_shape, feature_dim=512, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_stimuli, sequence_length, *stimulus_shape)`
                          where, by default, `nbr_stimuli=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param feature_dim: int defining the flatten number of dimension of the features for each stimulus. 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(Listener, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length

        # Multi-round:
        self._reset_rnn_states()

    def reset(self):
        self.apply(layer_init)
        
    def _reset_rnn_states(self):
        self.rnn_states = None 

    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        raise NotImplementedError

    def _reason(self, sentences, features):
        '''
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
        '''
        raise NotImplementedError
    
    def _utter(self, features, sentences):
        '''
        Reasons about the features and the listened sentences to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        raise NotImplementedError

    def forward(self, sentences, stimuli, multi_round=False, graphtype='straight_through_gumbel_softmax', tau=1.0):
        '''
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
        :param tau: 
        '''
        features = self._sense(stimuli=stimuli, sentences=sentences)
        decision_logits = self._reason(sentences=sentences, features=features)

        next_sentences_logits = None
        next_sentences = None
        if multi_round:
            next_sentences_logits, next_sentences = self._utter(features=features, sentences=sentences)

            if 'gumbel_softmax' in graphtype:
                straight_through = (graphtype == 'straight_through_gumbel_softmax')
                next_sentences = nn.functional.gumbel_softmax(logits=next_sentences_logits, tau=tau, hard=straight_through, dim=-1)
        else:
            self._reset_rnn_states()

        return decision_logits, next_sentences_logits, next_sentences 
