import torch
import torch.nn as nn


class Speaker(nn.Module):
    def __init__(self,obs_shape, feature_dim=512, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_stimuli, sequence_length, *stimulus_shape)`
                          where, by default, `nbr_stimuli=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param feature_dim: int defining the flatten number of dimension of the features for each stimulus. 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(Speaker, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length

        # Multi-round:
        self._reset_rnn_states()

    def _reset_rnn_states(self):
        self.rnn_states = None

    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `stimuli[:, 0]` is assumed as the target stimulus, while the others are distractors, if any. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''

        raise NotImplementedError

    def _utter(self, features, sentences=None):
        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        raise NotImplementedError

    def forward(self, stimuli, sentences=None, graphtype='straight_through_gumbel_softmax', tau=1.0, multi_round=False):
        '''
        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `stimuli[:,0]` is assumed as the target stimulus, while the others are distractors, if any. 
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution, following 
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
        :param tau: 
        '''

        # Add the target-boolean-channel:
        st_size = stimuli.size()
        batch_size = st_size[0]
        nbr_distractors_po = st_size[1]
        nbr_stimulus = st_size[2]

        target_channels = torch.zeros( batch_size, nbr_distractors_po, nbr_stimulus, 1, *(st_size[4:]))
        target_channels[:,0,...] = 1
        stimuli_target = torch.cat([stimuli, target_channels], dim=3)

        features = self._sense(stimuli=stimuli_target, sentences=sentences)
        next_sentences_logits, next_sentences = self._utter(features=features, sentences=sentences)

        if 'gumbel_softmax' in graphtype:    
            straight_through = (graphtype == 'straight_through_gumbel_softmax')
            next_sentences = nn.functional.gumbel_softmax(logits=next_sentences_logits, tau=tau, hard=straight_through, dim=-1)

        if not multi_round:
            self._reset_rnn_states()

        return next_sentences_logits, next_sentences 