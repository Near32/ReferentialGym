import torch
import torch.nn as nn

from ..networks import layer_init
from ..utils import gumbel_softmax

class Speaker(nn.Module):
    def __init__(self,obs_shape, vocab_size=100, max_sentence_length=10, agent_id='s0', logger=None):
        '''
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        '''
        super(Speaker, self).__init__()
        self.obs_shape = obs_shape
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.agent_id = agent_id
        self.logger = logger

        # Multi-round:
        self._reset_rnn_states()

    def reset(self):
        self.apply(layer_init)

    def _reset_rnn_states(self):
        self.rnn_states = None

    def clone(self, clone_id='s0'):
        logger = self.logger
        self.logger = None 
        clone = copy.deepcopy(self)
        clone.agent_id = clone_id
        clone.logger = logger 
        self.logger = logger  
        return clone 

    def save(self, path):
        logger = self.logger
        self.logger = None
        torch.save(self, path)
        self.logger = logger 
        
    def _log(self, log_dict):
        if self.logger is None: 
            return 

        for key, data in log_dict.items():
            logdict = { f"{self.agent_id}": {f"{key}":data}}
            self.logger.add_dict(logdict, batch=True)

    def _compute_tau(self, tau0):
        raise NotImplementedError

    def _sense(self, experiences, sentences=None):
        '''
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `experiences[:, 0]` is assumed as the target experience, while the others are distractors, if any. 
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
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        '''
        raise NotImplementedError

    def forward(self, experiences, sentences=None, graphtype='straight_through_gumbel_softmax', tau0=0.2, multi_round=False):
        '''
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `experiences[:,0]` is assumed as the target experience, while the others are distractors, if any. 
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution, following 
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: 
        '''

        # Add the target-boolean-channel:
        st_size = experiences.size()
        batch_size = st_size[0]
        nbr_distractors_po = st_size[1]
        nbr_stimulus = st_size[2]

        target_channels = torch.zeros( batch_size, nbr_distractors_po, nbr_stimulus, 1, *(st_size[4:]))
        target_channels[:,0,...] = 1
        if experiences.is_cuda: target_channels = target_channels.cuda()
        experiences_target = torch.cat([experiences, target_channels], dim=3)

        features = self._sense(experiences=experiences_target, sentences=sentences)
        next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = self._utter(features=features, sentences=sentences)
        
        if self.training:
            if 'gumbel_softmax' in graphtype:    
                self.tau = self._compute_tau(tau0=tau0)
                tau = self.tau.view((-1,1,1)).repeat(1,self.max_sentence_length,self.vocab_size)
                
                straight_through = (graphtype == 'straight_through_gumbel_softmax')
                next_sentences = gumbel_softmax(logits=next_sentences_logits, tau=tau, hard=straight_through, dim=-1)

        output_dict = {'sentences_widx':next_sentences_widx, 
                       'sentences_logits':next_sentences_logits, 
                       'sentences':next_sentences,
                       'features':features,
                       'temporal_features':temporal_features}
        
        if not multi_round:
            self._reset_rnn_states()

        self._log(output_dict)

        return output_dict