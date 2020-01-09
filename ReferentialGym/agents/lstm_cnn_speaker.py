import torch
import torch.nn as nn

from .speaker import Speaker
from ..networks import choose_architecture, layer_init, hasnan, BetaVAE


class LSTMCNNSpeaker(Speaker):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='s0', logger=None):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=0` (partial observability), and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        '''
        super(LSTMCNNSpeaker, self).__init__(obs_shape, vocab_size, max_sentence_length, agent_id, logger, kwargs)
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
        self.temporal_feature_encoder = layer_init(nn.LSTM(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=self.kwargs['dropout_prob'],
                                          bidirectional=False))

        assert(self.kwargs['symbol_processing_nbr_hidden_units'] == self.kwargs['temporal_encoder_nbr_hidden_units'])
        symbol_decoder_input_dim = self.kwargs['temporal_encoder_nbr_hidden_units']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)

        #self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_processing_nbr_hidden_units'], padding_idx=self.vocab_size)
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)

        self.tau_fc = layer_init(nn.Linear(self.kwargs['temporal_encoder_nbr_hidden_units'], 1 , bias=False))
        
        self.reset()
    
    def reset(self):
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None

    def _compute_tau(self, tau0, emb=None):
        if emb is None: emb = self.embedding_tf_final_outputs
        invtau = tau0 + torch.log(1+torch.exp(self.tau_fc(emb))).squeeze()
        return 1.0/invtau
        
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
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
        total_size = experiences.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            featout = self.cnn_encoder(stin)
            features.append(featout)
        features = torch.cat(features, dim=0)#.detach()
        features = features.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['cnn_encoder_feature_dim'])
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        return features 

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
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1, nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
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
        # (batch_size*(nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
        outputs = outputs.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['temporal_encoder_nbr_hidden_units'])
        # (batch_size, (nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
        
        # Taking only the target features: assumes partial observations anyway...
        # TODO: find a way to compute the sentence while attending other features in case of full observability...
        embedding_tf_final_outputs = outputs[:,0,-1,:].contiguous()
        # (batch_size, kwargs['temporal_encoder_feature_dim'])
        self.embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size, 1, -1)
        # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])
        
        #TODO : implement multi-round following below:
        '''
        if sentences is not None:
            # Consume the sentences:
            sentences = sentences.view((batch_size, -1))
            encoded_sentences = self.symbol_encoder(sentences).view((batch_size, self.max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])) 
            
            states = self.rnn_states
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            # Since we consume the sentence, rather than generating it, we prepend the encoded_sentences with ones:
            inputs = torch.ones((batch_size,1,self.kwargs['symbol_processing_nbr_hidden_units']))
            if encoded_sentences.is_cuda: inputs = inputs.cuda()
            encoded_sentences = torch.cat( [inputs, encoded_sentences], dim=1)
            # Then, as usual, we concatenate this sequence's vectors with repeated temporal feature embedding vectors:
            inputs = torch.cat( [embedding_tf_final_outputs.repeat(1,self.max_sentence_length+1,1), encoded_sentences], dim=-1)
            # (batch_size, max_sentence_length+1, (nbr_distractors+1)*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
            rnn_outputs, self.rnn_states = self.symbol_processing(inputs, states)          
            # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
            # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        '''

        vocab_stop_idx = self.vocab_size-1

        sentences_widx = [list() for _ in range(batch_size)]
        sentences_logits = [list() for _ in range(batch_size)]
        sentences_one_hots = [list() for _ in range(batch_size)]
        for b in range(batch_size):
            bemb = self.embedding_tf_final_outputs[b].view((1, 1, -1))
            # (batch_size=1, 1, kwargs['temporal_encoder_nbr_hidden_units'])
            init_rnn_state = bemb
            # (hidden_layer*num_directions=1, batch_size=1, 
            # kwargs['temporal_encoder_nbr_hidden_units']=kwargs['symbol_processing_nbr_hidden_units'])
            rnn_states = (init_rnn_state, torch.zeros_like(init_rnn_state))
            
            inputs = torch.zeros((1, 1, self.kwargs['symbol_processing_nbr_hidden_units']))
            if self.embedding_tf_final_outputs.is_cuda: inputs = inputs.cuda()
            # (batch_size=1, 1, kwargs['symbol_processing_nbr_hidden_units'])
            
            continuer = True
            sentence_token_count = 0
            while continuer:
                sentence_token_count += 1
                rnn_outputs, next_rnn_states = self.symbol_processing(inputs, rnn_states )
                # (batch_size=1, 1, kwargs['symbol_processing_nbr_hidden_units'])
                # (hidden_layer*num_directions, batch_size=1, kwargs['symbol_processing_nbr_hidden_units'])

                outputs = self.symbol_decoder(rnn_outputs.squeeze(1))
                # (batch_size=1, vocab_size)
                _, prediction = torch.softmax(outputs, dim=1).max(1)                        
                # (batch_size=1)
                prediction = prediction.unsqueeze(1).float()

                sentences_widx[b].append( prediction)
                sentences_logits[b].append( outputs.view((1,-1)))
                sentences_one_hots[b].append( nn.functional.one_hot(prediction.squeeze().long(), num_classes=self.vocab_size).view((1,-1)))
                
                # next inputs:
                inputs = self.symbol_encoder(outputs).unsqueeze(1)
                # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
                # next rnn_states:
                rnn_states = next_rnn_states
                
                stop_word_condition = (prediction == vocab_stop_idx)
                if len(sentences_widx[b]) >= self.max_sentence_length or stop_word_condition :
                    continuer = False 

            # Embed the sentence:
            # Padding token:
            while len(sentences_widx[b]) < self.max_sentence_length:
                sentences_widx[b].append((self.vocab_size-1)*torch.ones_like(prediction))

            sentences_widx[b] = torch.cat([ word_idx.view((1,1,-1)) for word_idx in sentences_widx[b]], dim=1)
            # (batch_size=1, sentence_length<=max_sentence_length, 1)
            sentences_logits[b] = torch.cat(sentences_logits[b], dim=0)
            # (sentence_length<=max_sentence_length, vocab_size)
            sentences_one_hots[b] = torch.cat(sentences_one_hots[b], dim=0) 
            # (sentence_length<=max_sentence_length, vocab_size)

        sentences_one_hots = nn.utils.rnn.pad_sequence(sentences_one_hots, batch_first=True, padding_value=0.0).float()
        # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)
        
        sentences_widx = torch.cat(sentences_widx, dim=0)
        # (batch_size, max_sentence_length, 1)
        if self.embedding_tf_final_outputs.is_cuda: sentences_widx = sentences_widx.cuda()


        return sentences_widx, sentences_logits, sentences_one_hots, self.embedding_tf_final_outputs.squeeze() 
        