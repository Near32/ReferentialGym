import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from .rnn_cnn_listener import RNNCNNListener
from ..networks import choose_architecture, layer_init, hasnan, BetaVAE


class MLPRNNCNNListener(RNNCNNListener):
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
        self.symbol_processing_nbr_hidden_units = kwargs['symbol_processing_nbr_hidden_units']
        super(MLPRNNCNNListener, self).__init__(
            kwargs=kwargs,
            obs_shape=obs_shape, 
            vocab_size=vocab_size, 
            max_sentence_length=max_sentence_length, 
            agent_id=agent_id, 
            logger=logger, 
            rnn_type=rnn_type
        )

        self.kwargs['symbol_processing_nbr_hidden_units'] = self.symbol_processing_nbr_hidden_units

        symbol_processing_input_dim = self.kwargs['symbol_embedding_size']
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

        decision_head_input_size = self.kwargs["symbol_processing_nbr_hidden_units"]+self.encoder_feature_shape
        self.decision_head = nn.Sequential(
            nn.Linear(decision_head_input_size,128),
            #nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=self.kwargs["dropout_prob"]),
            nn.ReLU(),
            nn.Linear(128, 2),
            #nn.Sigmoid()
        )
        self.decision_head.apply(layer_init)

    def reset_weights(self):
        #TODO: find a way to make this possible:
        # in spite of the mother classes needing this function too...
        #self.decision_head.apply(layer_init)
        super(MLPRNNCNNListener, self).reset_weights()


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
        if self.embedding_tf_final_outputs is None:
            if self.temporal_feature_encoder: 
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
                self.embedding_tf_final_outputs = self.normalization(embedding_tf_final_outputs.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
                self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape(batch_size, nbr_distractors_po, -1)
                # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
            else:
                self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
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
        bemb = self.embedding_tf_final_outputs.view((batch_size*nbr_distractors_po, -1))
        # (batch_size*nbr_distractors_po, cnn_encoder_feature_shape)
        
        
        for widx in range(rnn_outputs.size(1)):
            decision_inputs = rnn_outputs[:,widx,...].unsqueeze(1).repeat(1, nbr_distractors_po, 1)
            # (batch_size, nbr_distractors_po, kwargs['symbol_processing_nbr_hidden_units'])
            decision_inputs = decision_inputs.reshape(batch_size*nbr_distractors_po, -1)
            # (batch_size*nbr_distractors_po, kwargs['symbol_processing_nbr_hidden_units'])
            
            decision_head_input = torch.cat([decision_inputs, bemb], dim=-1)
            # (batch_size*nbr_distractors_po, 2*kwargs['symbol_processing_nbr_hidden_units'])
            
            decision_logits_until_widx = self.decision_head(decision_head_input).reshape((batch_size, nbr_distractors_po, 2))
            # Linear output...
            # (batch_size, nbr_distractors_po, 2)
                
            decision_logits.append(decision_logits_until_widx.unsqueeze(1))
            # (batch_size, 1, (nbr_distractors+1) 
            # / ? (descriptive mode depends on the role of the agent),
            # nodim / 2 )
        decision_logits = torch.cat(decision_logits, dim=1)
        # (batch_size, max_sentence_length, (nbr_distractors+1)
        # / ? (descriptive mode depends on the role of the agent),
        # nodim / 2 )
            
        if self.kwargs['descriptive']: # or kwargs is not None
            possible_targets = decision_logits[...,0]
            # (batch_size, max_sentence_length, (nbr_distractors+1), )
            not_target = decision_logits[...,1].max(dim=-1, keepdim=True)[0]
            # (batch_size, max_sentence_length, 1)                
            decision_logits = torch.cat([possible_targets, not_target], dim=-1 )
            # (batch_size, max_sentence_length, (nbr_distractors+2))
        
        # NOW: Regularization to make those values actual log probabilities...
        decision_logits = torch.log_softmax(decision_logits, dim=-1)
        
        return decision_logits, self.embedding_tf_final_outputs
