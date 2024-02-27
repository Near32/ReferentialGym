import torch
import torch.nn as nn

from .speaker import Speaker
from ..networks import layer_init

import copy


class RNNSpeaker(Speaker):
    def __init__(self, 
                    kwargs, 
                    obs_shape, 
                    vocab_size=100, 
                    max_sentence_length=10, 
                    agent_id='s0', 
                    logger=None,
                    rnn_type='lstm'):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=0` (partial observability), and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param rnn_type: String specifying the type of RNN to use, either 'gru' or 'lstm'.
        '''
        super(RNNSpeaker, self).__init__(
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs
        )

        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 
        self.force_eos = self.kwargs["force_eos"]

        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['symbol_processing_nbr_hidden_units'])
        
        symbol_decoder_input_dim = self.kwargs['symbol_embedding_size']
        self.rnn_type = rnn_type.lower()
        if 'lstm' in self.rnn_type:
            self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                batch_first=True,
                dropout=self.kwargs['dropout_prob'],
                bidirectional=False
            )
        elif 'gru' in self.rnn_type:
            self.symbol_processing = nn.GRU(input_size=symbol_decoder_input_dim,
                hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                batch_first=True,
                dropout=self.kwargs['dropout_prob'],
                bidirectional=False
            )
        else:
            raise NotImplementedError

        # SoS symbol is not part of the vocabulary as it is only prompting the RNNs
        # and is not part of the sentences being uttered.
        # TODO: when applying multi-round, it could be interesting to force SoS 
        # at the beginning of sentences so that agents can align rounds.
        self.sos_symbol = nn.Parameter(torch.zeros(1,1,self.config['symbol_embedding_size']))
        
        """
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
        self.symbol_encoder_dropout = nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
        """
        self.symbol_encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False),
            nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
        )
        # EoS symbol is part of the vocabulary:
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)


        self.tau_fc = nn.Sequential(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
                                          nn.Softplus())
        
        self.reset_weights()
    
    def reset_weights(self):
        # Reset EoS and SoS maybe?
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None

    def _compute_tau(self, tau0, h):
        tau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return tau

    def embed_sentences(self, sentences):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :returns embedded_sentences: Tensor of shape `(batch_size, max_sentence_length, symbol_embedding_size)` containing the padded sequence of embedded symbols.
        """
        batch_size = sentences.shape[0]
        # (batch_size, max_sentence_length, self.vocab_size)
        sentences = sentences.view((-1, self.vocab_size)).float()
        embedded_symbols = self.symbol_encoder(sentences) 
        # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
        embedded_sentences = embedded_symbols.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        return embedded_sentences

    def _sense(self, experiences, sentences=None):
        """
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, -1, nbr_stimulus, feature_dim).
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

    def _utter(self, features, sentences=None):
        '''
        TODO: update this description...
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1, nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
        # Forward pass:
        self.embedding_tf_final_outputs = self.normalization(features.reshape(-1, self.kwargs['symbol_processing_nbr_hidden_units']))
        self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, self.kwargs['nbr_distractors']+1, -1))
        # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])

        sentences_hidden_states = list()
        sentences_widx = list()
        sentences_logits = list()
        sentences_one_hots = list()
        
        bemb = self.embedding_tf_final_outputs.reshape((1, batch_size, -1))
        # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])
        init_rnn_state = bemb
        # (batch_size, hidden_layer*num_directions=1, 
        # kwargs['temporal_encoder_nbr_hidden_units']=kwargs['symbol_processing_nbr_hidden_units'])
        if 'lstm' in self.rnn_type:
            rnn_states = (init_rnn_state, torch.zeros_like(init_rnn_state))
        else:
            rnn_states = init_rnn_state

        # SoS token is given as initial input:
        '''
        # Assuming SoS is part of the vocabulary:
        inputs = self.symbol_encoder.weight[:, self.vocab_start_idx].reshape((1,1,-1))
        '''
        # Assuming SoS is not part of the vocabulary:
        inputs = self.sos_symbol.repeat(batch_size, 1, 1)
        #(batch_size, 1, self.kwargs['symbol_embedding_size']))
        if self.embedding_tf_final_outputs.is_cuda: inputs = inputs.cuda()
        
        max_sentence_length = self.max_sentence_length
        if self.force_eos:
            max_sentence_length -= 1

        for token_idx in range(max_sentence_length):
            rnn_outputs, next_rnn_states = self.symbol_processing(inputs, rnn_states )
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])

            sentences_hidden_states.append(rnn_outputs)
            
            outputs = self.symbol_decoder(rnn_outputs.reshape(batch_size,-1))
            # (batch_size, vocab_size)
            
            token_logits = torch.log_softmax(outputs, dim=-1)
            # (batch_size, vocab_size)
            token_dist = torch.distributions.Categorical(logits=token_logits)
            # (batch_size, vocab_size)
            
            if self.training:
                sampled_token = token_dist.sample().reshape(-1,1)
                # (batch_size,1)
            else:
                sampled_token = token_logits.argmax(dim=-1).reshape(-1,1)
                # (batch_size,1)
            
            sentences_widx.append(sampled_token)
            # (batch_size, 1)
            sentences_logits.append(token_logits)
            #sentences_logits.append(token_dist.log_probs(sampled_token))
            # (batch_size, vocab_size)
            
            token_one_hot = nn.functional.one_hot(sampled_token.long(), num_classes=self.vocab_size).view((-1, self.vocab_size))
            # (batch_size, vocab_size)
            sentences_one_hots.append(token_one_hot)
            # (batch_size, vocab_size)
            
            # Bookkeeping:
            ## next inputs:
            inputs = self.embed_sentences(token_one_hot.reshape(batch_size, 1, -1))
            # (batch_size, 1, kwargs['symbol_embedding_size'])
            ## next rnn_states:
            rnn_states = next_rnn_states
        
        if self.force_eos:
            eos_idx = self.vocab_stop_idx*torch.ones_like(sentences_widx[0])
            # (batch_size, 1)
            zeros = torch.zeros_like(sentences_logits[0])
            # (batch_size, vocab_size)
            sentences_widx.append(eos_idx)
            eos_one_hot = nn.functional.one_hot(eos_idx.view(-1,1).long(), num_classes=self.vocab_size).view((-1, self.vocab_size))
            # (batch_size, vocab_size)
            sentences_one_hots.append(eos_one_hot)
            sentences_logits.append(zeros)

            sentences_hidden_states.append(bemb.permute(1, 0, 2))
            

        sentences_hidden_states = torch.stack(sentences_hidden_states, dim=1)
        # (batch_size, max_sentence_length, kwargs['symbol_preprocessing_nbr_hidden_units'])
        sentences_widx = torch.stack(sentences_widx, dim=1)
        # (batch_size, max_sentence_length, 1)
        sentences_logits = torch.stack(sentences_logits, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        sentences_one_hots = torch.stack(sentences_one_hots, dim=1)
        # (batch_size, max_sentence_length, vocab_size)

        #if self.embedding_tf_final_outputs.is_cuda: sentences_widx = sentences_widx.cuda()


        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots, self.embedding_tf_final_outputs.squeeze() 
        
