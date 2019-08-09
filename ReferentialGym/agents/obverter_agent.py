import torch
import torch.nn as nn
import torch.nn.functional as F

from .listener import Listener
from ..networks import choose_architecture, layer_init


class ObverterAgent(Listener):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(ObverterAgent, self).__init__(obs_shape,vocab_size,max_sentence_length)
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        if self.kwargs['architecture'] == 'CNN':
            self.cnn_encoder = choose_architecture(architecture='CNN',
                                                  input_shape=cnn_input_shape,
                                                  hidden_units_list=None,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                  nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                                  kernels=self.kwargs['cnn_encoder_kernels'],
                                                  strides=self.kwargs['cnn_encoder_strides'],
                                                  paddings=self.kwargs['cnn_encoder_paddings'])
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
                                          dropout=0.0,
                                          bidirectional=False))

        symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=0.0,
                                      bidirectional=False)

        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)
        
        # Decision making: which input stimuli is the target? 
        decision_module_input = self.kwargs['symbol_processing_nbr_hidden_units'] + self.kwargs['temporal_encoder_nbr_hidden_units']
        self.decision_module = nn.Linear(decision_module_input, self.kwargs['nbr_distractors']+1)
        
        self.reset()

    def reset(self):
        self.symbol_processing.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.decision_module.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None

    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        batch_size = stimuli.size(0)
        stimuli = stimuli.view(-1, *(stimuli.size()[3:]))
        features = []
        total_size = stimuli.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(stimuli, split_size_or_sections=mini_batch_size, dim=0):
            features.append( self.cnn_encoder(stin))
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, *(self.obs_shape[:2]), -1)
        # (batch_size, nbr_distractors+1, nbr_stimulus, feature_dim)
        return features 

    def _reason(self, sentences, features):
        '''
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
        '''
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
        
        embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
        self.embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size, -1)
        # (batch_size, (nbr_distractors+1) * kwargs['temporal_encoder_nbr_hidden_units'])
        
        # Consume the sentences:
        sentences = sentences.view((-1, self.vocab_size))
        embedded_sentences = self.symbol_encoder(sentences).view((batch_size, self.max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units']))
        states = self.rnn_states
        # (batch_size, kwargs['max_sentence_length'], kwargs['symbol_processing_nbr_hidden_units'])
        rnn_outputs, self.rnn_states = self.symbol_processing(embedded_sentences, states)          
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        # Compute the decision: following the last hidden/output vector from the rnn:
        decision_inputs = rnn_outputs[:,-1,...]
        # (batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        decision_logits = []
        for b in range(batch_size):
            bemb = self.embedding_tf_final_outputs[b].view((self.obs_shape[0], -1))
            # ( (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
            bdin = decision_inputs[b].unsqueeze(1)
            # (kwargs['symbol_processing_nbr_hidden_units'], 1)
            dl = torch.matmul( bemb, bdin).squeeze()
            # ( (nbr_distractors+1), )
            decision_logits.append(dl.unsqueeze(0))
        decision_logits = torch.cat(decision_logits, dim=0)
        # (batch_size, (nbr_distractors+1) )
        return decision_logits


    def _utter(self, features, sentences):
        '''
        Reasons about the features and the listened sentences to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''

        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1, nbr_stimulus, feature_dim)
        
        if self.embedding_tf_final_outputs is None: 
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
            
            embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
            self.embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size, -1)
            # (batch_size, (nbr_distractors+1) * kwargs['temporal_encoder_nbr_hidden_units'])
            
        # No need to consume the sentences:
        # it has been consumed already in the _reason function.
        # self.rnn_states contains all the information about it.
        # Or, else, self.rnn_states = None and we start anew:

        # The operation (max/min) to use during the computation of the sentences
        # depends on the current role of the agent, that is determined by 
        # `sentences==None` ==> Speaker (first round).
        # TODO: account for multi-round communication...
        if sentences is None:   operation = torch.max 
        else:   operation = torch.min 

        # Similarly, as a speaker/teacher, it is assumed that `target_idx=0`.
        # TODO: decide on a strategy for the listener/student's predicted_target_idx's argument...
        predicted_target_idx = torch.zeros((batch_size, )).long()
        if self.embedding_tf_final_outputs.is_cuda: predicted_target_idx = predicted_target_idx.cuda()

        # Utter the next sentences:
        next_sentences_idx, \
        next_sentences_logits, \
        next_sentences_one_hots = ObverterAgent._compute_sentence(features_embedding=self.embedding_tf_final_outputs,
                                                         target_idx=predicted_target_idx,
                                                         symbol_encoder=self.symbol_encoder,
                                                         symbol_processing=self.symbol_processing,
                                                         symbol_decoder=self.symbol_decoder,
                                                         init_rnn_states=self.rnn_states,
                                                         vocab_size=self.vocab_size,
                                                         max_sentence_length=self.max_sentence_length,
                                                         nbr_distractors_po=self.kwargs['nbr_distractors']+1,
                                                         operation=operation)

        return next_sentences_logits, next_sentences_one_hots

    def _compute_sentence(features_embedding, 
                          target_idx, 
                          symbol_encoder, 
                          symbol_processing, 
                          symbol_decoder, 
                          init_rnn_states=None, 
                          vocab_size=10, 
                          max_sentence_length=14,
                          nbr_distractors_po=1,
                          operation=torch.max,
                          vocab_stop_idx=0):
        '''
        Compute sentences using the obverter approach, adapted to referential game variants following the
        descriptive approach described in the work of [Choi et al., 2018](http://arxiv.org/abs/1804.02341).

        In descriptive move, `nbr_distractors_po=1` and `target_idx=torch.zeros((batch_size,1))`, 
        thus the algorithm behaves exactly like in Choi et al. (2018).
        Otherwise, the the likelyhoods for the target stimulus of being chosen by the decision module 
        is considered solely and the algorithm aims at maximizing/minimizing (following :param operation:) 
        this likelyhood over the sentence's next word.
        
        :param features_embedding: Tensor of (temporal) features embedding of shape `(batch_size, *self.obs_shape)`.
        :param target_idx: Tensor of indices of the target stimuli of shape `(batch_size, 1)`.
        :param symbol_encoder: torch.nn.Module used to embed vocabulary indices into vocabulary embeddings.
        :param symbol_processing: torch.nn.Module used to generate the sentences.
        :param symbol_decoder: torch.nn.Module used to decode the embeddings generated by the `:param symbol_processing:` module. 
        :param init_rnn_states: None or Tuple of Tensors to initialize the symbol_processing's rnn states.
        :param vocab_size: int, size of the vocabulary.
        :param max_sentence_length: int, maximal length for each generated sentences.
        :param nbr_distractors_po: int, number of distractors and target, i.e. `nbr_distractors+1.
        :param operation: Function, expect `torch.max` or `torch.min`.
        :param vocab_stop_idx: int, index of the STOP symbol in the vocabulary.

        :returns:
            - sentences_idx: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], 1)` where `b` is the batch index.
                             It represents the indices of the chosen words.
            - sentences_logits: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the logits of words over the decision module's potential to choose the target stimulus as output.
            - sentences_one_hots: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the sentences as one-hot-encoded word vectors.
        '''
        batch_size = features_embedding.size(0)
        states = init_rnn_states
        vocab_idx = torch.zeros((vocab_size,vocab_size))
        for i in range(vocab_size): vocab_idx[i,i] = 1
        if features_embedding.is_cuda: vocab_idx = vocab_idx.cuda()
        vocab_idx = symbol_encoder(vocab_idx).unsqueeze(1)
        # (batch_size=vocab_size, 1, kwargs['symbol_processing_nbr_hidden_units'])

        sentences_idx = [list() for _ in range(batch_size)]
        sentences_logits = [list() for _ in range(batch_size)]
        sentences_one_hots = [list() for _ in range(batch_size)]
        for b in range(batch_size):
            bemb = features_embedding[b].view((nbr_distractors_po, -1))
            # ( (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
            btarget_idx = target_idx[b]
            # (1,)
            continuer = True
            while continuer:
                if states is not None:
                    hs, cs = states[0], states[1]
                    hs = hs.repeat( 1, vocab_size, 1)
                    cs = cs.repeat( 1, vocab_size, 1)
                    rnn_states = (hs, cs)
                else :
                    rnn_states = states

                rnn_outputs, next_rnn_states = symbol_processing(vocab_idx, rnn_states )
                # (batch_size=vocab_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
                # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
                
                # Compute the decision: following the last hidden/output vector from the rnn:
                decision_inputs = rnn_outputs[:,-1,...]
                # (batch_size=vocab_size, kwargs['symbol_processing_nbr_hidden_units'])
                decision_logits = []
                for bv in range(vocab_size):
                    bdin = decision_inputs[bv].unsqueeze(1)
                    # (kwargs['symbol_processing_nbr_hidden_units'], 1)
                    dl = torch.matmul( bemb, bdin).view((1,-1))
                    # ( 1, (nbr_distractors+1))
                    decision_logits.append(dl)
                decision_logits = torch.cat(decision_logits, dim=0)
                # (batch_size=vocab_size, (nbr_distractors+1) )
                
                decision_probs = F.softmax(decision_logits, dim=1)
                # (batch_size=vocab_size, (nbr_distractors+1) )
                target_decision_probs_per_vocab = decision_probs[:,btarget_idx].cpu()
                # (batch_size=vocab_size, )
                _, vocab_idx_argop = operation(target_decision_probs_per_vocab, dim=0)
                # (batch_size=vocab_size, )
                
                sentences_idx[b].append( vocab_idx_argop)
                sentences_logits[b].append( target_decision_probs_per_vocab.view((1,1,-1)))
                sentences_one_hots[b].append( nn.functional.one_hot(vocab_idx_argop, num_classes=vocab_size).view((1,-1)))
                
                # next rnn_states:
                states = [st[-1, vocab_idx_argop].view((1,1,-1)) for st in next_rnn_states]

                if len(sentences_idx[b]) >= max_sentence_length or vocab_idx_argop == vocab_stop_idx:
                    continuer = False 

            # Embed the sentence:
            sentences_idx[b] = torch.cat([ torch.FloatTensor([word_idx]).view((1,1,-1)) for word_idx in sentences_idx[b]], dim=1)
            # (batch_size=1, sentence_length<=max_sentence_length, 1)
            sentences_logits[b] = torch.cat(sentences_logits[b], dim=1)
            # (batch_size=1, sentence_length<=max_sentence_length, vocab_size)
            sentences_one_hots[b] = torch.cat(sentences_one_hots[b], dim=0) 
            # (batch_size=1, sentence_length<=max_sentence_length, vocab_size)

            # Reset the state for the next sentence generation in the batch:
            states = init_rnn_states

        sentences_one_hots = nn.utils.rnn.pad_sequence(sentences_one_hots, batch_first=True, padding_value=0.0).float()
        # (batch_size=1, max_sentence_length<=max_sentence_length, vocab_size)

        if features_embedding.is_cuda: sentences_one_hots = sentences_one_hots.cuda()

        return sentences_idx, sentences_logits, sentences_one_hots
