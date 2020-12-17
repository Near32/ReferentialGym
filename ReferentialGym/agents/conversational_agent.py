import torch
import torch.nn as nn

from ..networks import layer_init
from ..utils import gumbel_softmax
from .agent import Agent 
from .speaker import sentence_length_logging_hook, entropy_logging_hook, entropy_regularization_loss_hook, mdl_principle_loss_hook, oov_loss_hook


class ConversationalAgent(Agent):
    def __init__(self, role, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='s0', logger=None, kwargs=None):
        '''
        :param role: String defining the role of the agent.
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        '''
        super(ConversationalAgent, self).__init__(agent_id=agent_id, 
                                                  obs_shape=obs_shape,
                                                  vocab_size=vocab_size,
                                                  max_sentence_length=max_sentence_length,
                                                  logger=logger, 
                                                  kwargs=kwargs,
                                                  role=role)
        
        
        self.register_hook(sentence_length_logging_hook)
        self.register_hook(entropy_logging_hook)

        if 'with_speaker_entropy_regularization' in self.kwargs \
         and self.kwargs['with_speaker_entropy_regularization']:
            self.register_hook(entropy_regularization_loss_hook)

        if 'with_mdl_principle' in self.kwargs \
         and self.kwargs['with_mdl_principle']:
            self.register_hook(mdl_principle_loss_hook)

        if ('with_utterance_penalization' in self.kwargs or 'with_utterance_promotion' in self.kwargs) \
         and (self.kwargs['with_utterance_penalization'] or self.kwargs['with_utterance_promotion']):
            self.register_hook(oov_loss_hook)

        self._reset_inner_state()

    def reset(self):
        self.apply(layer_init)

    def _reset_inner_state(self):
        self.inner_state = None
        self.embedding_tf_final_outputs = None

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

    def forward(self, experiences, sentences=None, multi_round=False, graphtype='straight_through_gumbel_softmax', tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                            `experiences[:,0]` is assumed as the target experience, while the others are distractors, if any. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        self.multi_round = multi_round
        self._sense(experiences=experiences, sentences=sentences)
        reasoning_output = self._reason(sentences=sentences, features=self.features)
        
        utter_outputs = self._utter(features=self.features, sentences=sentences)
        if len(utter_outputs) == 5:
            next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        else:
            next_sentences_hidden_states = None
            next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        
        if self.training:
            if 'gumbel_softmax' in graphtype:    
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

        output_dict = {
            'output':reasoning_output,
            'sentences_widx':next_sentences_widx, 
            'sentences_logits':next_sentences_logits, 
            'sentences_one_hot':next_sentences,
            #'features':features,
            'temporal_features':temporal_features
        }

        if not multi_round:
            self._reset_inner_state()

        return output_dict