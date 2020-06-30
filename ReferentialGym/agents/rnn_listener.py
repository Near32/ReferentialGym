import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminative_listener import DiscriminativeListener
from ..networks import layer_init


class RNNListener(DiscriminativeListener):
    def __init__(self,
                    kwargs, 
                    obs_shape, 
                    vocab_size=100, 
                    max_sentence_length=10, 
                    agent_id='l0', 
                    logger=None,
                    rnn_type='lstm'):
        """
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param rnn_type: String specifying the type of RNN to use, either 'gru' or 'lstm'.
        """
        super(RNNListener, self).__init__(
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs
        )
        
        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 

        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['symbol_processing_nbr_hidden_units'])
    
        symbol_processing_input_dim = self.kwargs['symbol_embedding_size']
        self.rnn_type = rnn_type.lower()
        if 'lstm' in self.rnn_type:
            self.symbol_processing = nn.LSTM(input_size=symbol_processing_input_dim,
                                          hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                          num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=self.kwargs['dropout_prob'],
                                          bidirectional=False)
        elif 'gru' in self.rnn_type:
            self.symbol_processing = nn.GRU(input_size=symbol_processing_input_dim,
                                          hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                          num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=self.kwargs['dropout_prob'],
                                          bidirectional=False)
        else:
            raise NotImplementedError
        '''
        self.symbol_processing_learnable_initial_state = nn.Parameter(
                torch.zeros(1,1,self.kwargs['symbol_processing_nbr_hidden_units'])
        )
        '''
        #self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_processing_nbr_hidden_units'], padding_idx=self.vocab_size)
        #self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
        
        self.symbol_encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False),
            nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
            )
        
        self.tau_fc = nn.Sequential(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
                                          nn.Softplus())

        '''
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, self.kwargs['max_sentence_length'], 1)))
        '''
        
        self.projection_normalization = None #nn.BatchNorm1d(num_features=self.kwargs['max_sentence_length']*self.kwargs['symbol_processing_nbr_hidden_units'])

        self.reset()

    def reset(self):
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        """
        Called at the agent level at the end of the `compute` function.
        """
        self.embedding_tf_final_outputs = None

    def _compute_tau(self, tau0, h):
        '''
        invtau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return invtau
        '''
        raise NotImplementedError

    def _sense(self, experiences, sentences=None):
        r"""
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, -1, feature_dim).
        
        """
        batch_size = experiences.size(0)
        nbr_distractors_po = experiences.size(1)
        nbr_stimulus = experiences.size(2)
        
        self.features = experiences.reshape(
            batch_size,
            nbr_distractors_po,
            nbr_stimulus,
            -1, 
        )
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        return self.features

    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences:   Tensor of shape `(batch_size, max_sentence_length, vocab_size)` 
                            containing the padded sequence of (potentially one-hot-encoded) symbols.
                            NOTE: max_sentence_length may be different from self.max_sentence_lenght 
                            as the padding is padding by batch and only care about the maximal 
                            sentence length of said batch.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        batch_size = features.size(0)
        nbr_distractors_po = features.size(1)        
        # (batch_size, nbr_distractors+1, nbr_stimulus, feature_dim)
        # Forward pass:
        self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, self.kwargs['symbol_processing_nbr_hidden_units'])))
        self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, nbr_distractors_po, -1))
        # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])

        # Consume the sentences:
        # (batch_size, max_sentence_length, self.vocab_size)
        sentences = sentences.view((-1, self.vocab_size))
        encoded_symbols = self.symbol_encoder(sentences) 
        # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
        encoded_sentences = encoded_symbols.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        
        # We initialize the rnn_states to either None, if it is not multi-round, or:
        states = self.rnn_states
        rnn_outputs, self.rnn_states = self.symbol_processing(encoded_sentences, states)          
        '''
        init_rnn_state = self.symbol_processing_learnable_initial_state.expand(
            batch_size,
            -1,
            -1
        ) 
        rnn_states = (
            init_rnn_state,
            torch.zeros_like(init_rnn_state)
        )
        '''
        
        """
        rnn_states = None
        rnn_outputs, next_rnn_states = self.symbol_processing(encoded_sentences, rnn_states)          
        """

        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        # Batch Normalization:
        if self.projection_normalization is not None:
            rnn_outputs = self.projection_normalization(rnn_outputs.reshape((batch_size, -1)))
            rnn_outputs = rnn_outputs.reshape((batch_size, -1, self.kwargs['symbol_processing_nbr_hidden_units']))

        # Compute the decision: following each hidden/output vector from the rnn:
        decision_logits = []
        for widx in range(rnn_outputs.size(1)):
            decision_inputs = rnn_outputs[:,widx,...]
            # (batch_size, kwargs['symbol_processing_nbr_hidden_units'])
            decision_logits_until_widx = []
            for b in range(batch_size):
                bemb = self.embedding_tf_final_outputs[b]
                # ( (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent), 
                # kwargs['temporal_encoder_nbr_hidden_units']==kwargs['symbol_processing_nbr_hidden_units'])
                bdin = decision_inputs[b].unsqueeze(1)
                # (kwargs['symbol_processing_nbr_hidden_units'], 1)
                dl = torch.matmul( bemb, bdin).view((1,-1))
                # ( 1, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent))
                decision_logits_until_widx.append(dl)
            decision_logits_until_widx = torch.cat(decision_logits_until_widx, dim=0)
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            decision_logits.append(decision_logits_until_widx.unsqueeze(1))
            # (batch_size, 1, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
        decision_logits = torch.cat(decision_logits, dim=1)
        # (batch_size, max_sentence_length, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )           

        #TODO: why would this be needed already?? Apparently in case of descriptive mode, cf obverter...
        '''
        not_target_logit = self.not_target_logits_per_token.repeat(batch_size, 1, 1)
        if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
        decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
        # (batch_size, (nbr_distractors+1) )
        '''

        return decision_logits, self.embedding_tf_final_outputs


    def _utter(self, features, sentences):
        """
        Reasons about the features and the listened sentences to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """

        """
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        """
        raise NotImplementedError
