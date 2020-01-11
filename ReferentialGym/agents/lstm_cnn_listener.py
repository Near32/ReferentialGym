import torch
import torch.nn as nn
import torch.nn.functional as F

from .listener import Listener
from ..networks import choose_architecture, layer_init


class LSTMCNNListener(Listener):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='l0', logger=None):
        """
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        """
        super(LSTMCNNListener, self).__init__(obs_shape, vocab_size, max_sentence_length, agent_id, logger, kwargs)
        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        if self.kwargs['architecture'] == 'CNN':
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  hidden_units_list=None,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                  nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                                  kernels=self.kwargs['cnn_encoder_kernels'],
                                                  strides=self.kwargs['cnn_encoder_strides'],
                                                  paddings=self.kwargs['cnn_encoder_paddings'],
                                                  dropout=self.kwargs['dropout_prob'])
        elif 'ResNet18' in self.kwargs['architecture']:
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'])

        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        self.temporal_feature_encoder = layer_init(
                                        nn.LSTM(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=self.kwargs['dropout_prob'],
                                          bidirectional=False))

        assert(self.kwargs['symbol_processing_nbr_hidden_units'] == self.kwargs['temporal_encoder_nbr_hidden_units'])
        symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)

        #self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_processing_nbr_hidden_units'], padding_idx=self.vocab_size)
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)
        
        self.tau_fc = layer_init(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1 , bias=False))
        
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, self.kwargs['max_sentence_length'], 1)))
        self.register_parameter(name='not_target_logits_per_token', param=self.not_target_logits_per_token)

        self.reset()

    def reset(self):
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None

    def _compute_tau(self, tau0):
        invtau = tau0 + torch.log(1+torch.exp(self.tau_fc(self.rnn_states[0][-1]))).squeeze()
        return 1.0/invtau
        
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
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
        total_size = experiences.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            features.append( self.cnn_encoder(stin))
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['cnn_encoder_feature_dim'])
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        return features 

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
        # (batch_size, nbr_distractors+1, nbr_stimulus, feature_dim)
        # Forward pass:
        features = features.view(-1, *(features.size()[2:]))
        # (batch_size*(nbr_distractors+1), nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
        rnn_outputs = []
        total_size = features.size(0)
        mini_batch_size = min(self.kwargs['temporal_encoder_mini_batch_size'], total_size)
        for featin in torch.split(features, split_size_or_sections=mini_batch_size, dim=0):
            outputs, _ = self.temporal_feature_encoder(featin)
            rnn_outputs.append( outputs)
        outputs = torch.cat(rnn_outputs, dim=0)
        outputs = outputs.view(batch_size, *(self.obs_shape[:2]), -1)
        
        # Caring only about the final output:
        embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
        # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
        self.embedding_tf_final_outputs = embedding_tf_final_outputs
        #self.embedding_tf_final_outputs = torch.sigmoid(embedding_tf_final_outputs)
        # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
        
        # Consume the sentences:
        # (batch_size, max_sentence_length, self.vocab_size)
        sentences = sentences.view((-1, self.vocab_size))
        encoded_symbols = self.symbol_encoder(sentences) 
        # (batch_size*max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])
        encoded_sentences = encoded_symbols.view((batch_size, -1, self.kwargs['symbol_processing_nbr_hidden_units']))
        # (batch_size, max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])
        
        # We initialize the rnn_states to either None, if it is not multi-round, or:
        # TODO: find a strategy for multiround...
        states = self.rnn_states
        rnn_outputs, self.rnn_states = self.symbol_processing(encoded_sentences, states)          
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        # Compute the decision: following each hidden/output vector from the rnn:
        decision_logits = []

        #rnn_outputs = torch.sigmoid(rnn_outputs)
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

        #TODO: why would this be needed already??
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