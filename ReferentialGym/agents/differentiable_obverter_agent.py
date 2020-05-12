import torch
import torch.nn as nn
import torch.nn.functional as F

from .listener import Listener
from ..networks import choose_architecture, layer_init, BetaVAE
from ..utils import gumbel_softmax


class DifferentiableObverterAgent(Listener):
    def __init__(self,
                 kwargs, 
                 obs_shape, 
                 vocab_size=100, 
                 max_sentence_length=10, 
                 agent_id='o0', 
                 logger=None, 
                 use_sentences_one_hot_vectors=True,
                 differentiable=True):
        """
        :param obs_shape: tuple defining the shape of the experience following `(nbr_distractors+1, nbr_stimulus, *experience_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param use_sentences_one_hot_vectors: Boolean specifying whether to use (potentially ST-GS) one-hot-encoded
            vector sentences as input (then consumable by a nn.Liner layer for the embedding, instead of nn.Embedding),
            or to use word/token indices sentences that requires two differentiation trick (from the speaker 
            upon production and from the listener upon consumption).
        :param differentiable: Boolean specifying whether to use the differentiable graph (from loss to speaker via listener),
            or the non-differentiable graph, only updating the listener.
        """
        super(DifferentiableObverterAgent, self).__init__(
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs)
        
        self.kwargs = kwargs 

        # Differentiability?
        self.differentiable = differentiable
        if not(self.differentiable):

            del self.input_stream_ids['speaker']['modules:current_listener:sentences_one_hot']
            del self.input_stream_ids['speaker']['modules:current_listener:sentences_widx']

            del self.input_stream_ids['listener']['modules:current_speaker:sentences_one_hot']
            del self.input_stream_ids['listener']['modules:current_speaker:sentences_widx']

            self.input_stream_ids['speaker'].update({
                'modules:current_listener:sentences_one_hot.detach':'sentences_one_hot',
                'modules:current_listener:sentences_widx.detach':'sentences_widx',
            })

            self.input_stream_ids['listener'].update({
                'modules:current_speaker:sentences_one_hot.detach':'sentences_one_hot',
                'modules:current_speaker:sentences_widx.detach':'sentences_widx', 
            })
        

        self.use_sentences_one_hot_vectors = use_sentences_one_hot_vectors
        self.use_learning_not_target_logit = True

        cnn_input_shape = self.obs_shape[2:]
        MHDPANbrHead=4
        MHDPANbrRecUpdate=1
        MHDPANbrMLPUnit=512
        MHDPAInteractionDim=128
        if 'mhdpa_nbr_head' in self.kwargs: MHDPANbrHead = self.kwargs['mhdpa_nbr_head']
        if 'mhdpa_nbr_rec_update' in self.kwargs: MHDPANbrRecUpdate = self.kwargs['mhdpa_nbr_rec_update']
        if 'mhdpa_nbr_mlp_unit' in self.kwargs: MHDPANbrMLPUnit = self.kwargs['mhdpa_nbr_mlp_unit']
        if 'mhdpa_interaction_dim' in self.kwargs: MHDPAInteractionDim = self.kwargs['mhdpa_interaction_dim']
        
        self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                               kwargs=self.kwargs,
                                               input_shape=cnn_input_shape,
                                               feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                               nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                               kernels=self.kwargs['cnn_encoder_kernels'],
                                               strides=self.kwargs['cnn_encoder_strides'],
                                               paddings=self.kwargs['cnn_encoder_paddings'],
                                               fc_hidden_units_list=self.kwargs['cnn_encoder_fc_hidden_units'],
                                               dropout=self.kwargs['dropout_prob'],
                                               MHDPANbrHead=MHDPANbrHead,
                                               MHDPANbrRecUpdate=MHDPANbrRecUpdate,
                                               MHDPANbrMLPUnit=MHDPANbrMLPUnit,
                                               MHDPAInteractionDim=MHDPAInteractionDim)
        
        self.use_feat_converter = self.kwargs['use_feat_converter'] if 'use_feat_converter' in self.kwargs else False 
        if self.use_feat_converter:
            self.feat_converter_input = self.cnn_encoder.get_feature_shape()

        if 'BetaVAE' in self.kwargs['architecture'] or 'MONet' in self.kwargs['architecture']:
            self.VAE_losses = list()
            self.compactness_losses = list()
            self.buffer_cnn_output_dict = dict()
            
            # N.B: with a VAE, we want to learn the weights in any case:
            if 'agent_learning' in self.kwargs:
                assert('transfer_learning' not in self.kwargs['agent_learning'])
            
            self.vae_detached_featout = False
            if self.kwargs['vae_detached_featout']:
                self.vae_detached_featout = True

            self.VAE = self.cnn_encoder

            self.use_feat_converter = True
            self.feat_converter_input = self.cnn_encoder.latent_dim
        else:
            if 'agent_learning' in self.kwargs and 'transfer_learning' in self.kwargs['agent_learning']:
                self.cnn_encoder.detach_conv_maps = True

        self.encoder_feature_shape = self.cnn_encoder.get_feature_shape()
        if self.use_feat_converter:
            self.featout_converter = []
            self.featout_converter.append(nn.Linear(self.feat_converter_input, self.kwargs['cnn_encoder_feature_dim']*2))
            self.featout_converter.append(nn.ReLU())
            self.featout_converter.append(nn.Linear(self.kwargs['cnn_encoder_feature_dim']*2, self.kwargs['feat_converter_output_size'])) 
            self.featout_converter.append(nn.ReLU())
            self.featout_converter =  nn.Sequential(*self.featout_converter)
            self.encoder_feature_shape = self.kwargs['feat_converter_output_size']
        
        self.cnn_encoder_normalization = nn.BatchNorm1d(num_features=self.encoder_feature_shape)

        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        if self.kwargs['temporal_encoder_nbr_rnn_layers'] > 0:
            self.temporal_feature_encoder = layer_init(
                nn.GRU(input_size=temporal_encoder_input_dim,
                      hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                      num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                      batch_first=True,
                      dropout=self.kwargs['dropout_prob'],
                      bidirectional=False)
                )
        else:
            self.temporal_feature_encoder = None
            print("WARNING: Symbol processing :: the number of hidden units is being reparameterized to fit to convolutional features.")
            self.kwargs['temporal_encoder_nbr_hidden_units'] = self.kwargs['nbr_stimulus']*self.encoder_feature_shape
            self.kwargs['symbol_processing_nbr_hidden_units'] = self.kwargs['temporal_encoder_nbr_hidden_units']

        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['temporal_encoder_nbr_hidden_units'])
        
        #symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']
        symbol_decoder_input_dim = self.kwargs['symbol_embedding_size']
        self.symbol_processing = nn.GRU(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)

        '''
        if self.use_sentences_one_hot_vectors:
            #self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
            self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
        else:
            #self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_processing_nbr_hidden_units'], padding_idx=self.vocab_size)
            self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_embedding_size'], padding_idx=self.vocab_size)
        
        self.symbol_decoder = nn.ModuleList()
        self.symbol_decoder.append(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size))
        if self.kwargs['dropout_prob']: self.symbol_decoder.append(nn.Dropout(p=self.kwargs['dropout_prob']))
        
        self.tau_fc = layer_init(nn.Linear(self.kwargs['temporal_encoder_nbr_hidden_units'], 1 , bias=False))
        
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1,self.kwargs['max_sentence_length'])))
        self.register_parameter(name='not_target_logits_per_token', param=self.not_target_logits_per_token)
        '''

        '''
        self.symbol_encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False),
            nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
            )
        '''
        if self.use_sentences_one_hot_vectors:
            self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
        else:
            self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_embedding_size'], padding_idx=self.vocab_size)
        
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)
        
        self.tau_fc = nn.Sequential(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
                                          nn.Softplus())

        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, self.kwargs['max_sentence_length'], 1)))
        self.register_parameter(name='not_target_logits_per_token', param=self.not_target_logits_per_token)

        self.projection_normalization = None #nn.BatchNorm1d(num_features=self.kwargs['max_sentence_length']*self.kwargs['symbol_processing_nbr_hidden_units'])

        self.reset()

    def reset(self):
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None
        
        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

    def _compute_tau(self, tau0, h):
        invtau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return invtau

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
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
        feat_maps = []
        total_size = experiences.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            if isinstance(self.cnn_encoder, BetaVAE):
                cnn_output_dict  = self.cnn_encoder.compute_loss(stin)
                if 'VAE_loss' in cnn_output_dict:
                    self.VAE_losses.append(cnn_output_dict['VAE_loss'])
                
                if hasattr(self.cnn_encoder, 'compactness_losses') and self.cnn_encoder.compactness_losses is not None:
                    self.compactness_losses.append(self.cnn_encoder.compactness_losses.cpu())
                
                for key in cnn_output_dict:
                    if key not in self.buffer_cnn_output_dict:
                        self.buffer_cnn_output_dict[key] = list()
                    self.buffer_cnn_output_dict[key].append(cnn_output_dict[key].cpu())

                if self.kwargs['vae_use_mu_value']:
                    featout = self.cnn_encoder.mu 
                else:
                    featout = self.cnn_encoder.z

                if self.vae_detached_featout:
                    featout = featout.detach()

                featout = self.featout_converter(featout)

                feat_map = self.cnn_encoder.get_feat_map()
            else:
                featout = self.cnn_encoder(stin)
                if self.use_feat_converter:
                    featout = self.featout_converter(featout)

                feat_map = self.cnn_encoder.get_feat_map()
            
            features.append(featout)
            feat_maps.append(feat_map)

        self.features = self.cnn_encoder_normalization(torch.cat(features, dim=0))
        self.feat_maps = torch.cat(feat_maps, dim=0)
        
        self.features = self.features.view(batch_size, nbr_distractors_po, self.config['nbr_stimulus'], -1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = torch.cat(self.VAE_losses).contiguous()#.view((batch_size,-1)).mean(dim=-1)
            
            for key in self.buffer_cnn_output_dict:
                self.log_dict[key] = torch.cat(self.buffer_cnn_output_dict[key]).mean()

            self.log_dict['kl_capacity'] = torch.Tensor([100.0*self.cnn_encoder.EncodingCapacity/self.cnn_encoder.maxEncodingCapacity])
            if len(self.compactness_losses):
                self.log_dict['unsup_compactness_loss'] = torch.cat(self.compactness_losses).mean()

        return self.features

    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        # Forward pass:
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
            self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape(batch_size, self.kwargs['nbr_distractors']+1, -1)
            # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
        else:
            self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
            self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, self.kwargs['nbr_distractors']+1, -1))
            # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])

        # Consume the sentences:
        if self.use_sentences_one_hot_vectors:
            sentences = sentences.view((-1, self.vocab_size))
            # (batch_size*max_sentence_length, vocab_size)
            embedded_sentences = self.symbol_encoder(sentences)
            embedded_sentences = embedded_sentences.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
            # (batch_size, max_sentence_length, kwargs['symbol_embedding_size'])
        else:
            sentences = sentences.view((batch_size, -1))
            # (batch_size, max_sentence_length)
            embedded_sentences = self.symbol_encoder(sentences.long())
            if self.differentiable:
                embedded_sentences = (sentences-sentences.detach()).unsqueeze(-1)+embedded_sentences
            embedded_sentences = embedded_sentences.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
            # (batch_size, max_sentence_length, kwargs['symbol_embedding_size'])
        
        # We initialize the rnn_states to either None, if it is not multi-round, or:
        # TODO: find a strategy for multiround...
        states = self.rnn_states
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        rnn_outputs, self.rnn_states = self.symbol_processing(embedded_sentences, states)          
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        '''
        TODO: find out whether this projection normalization is necessary:
        '''
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
                bemb = self.embedding_tf_final_outputs[b].view((-1, self.kwargs['temporal_encoder_nbr_hidden_units']))
                # ( (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent), kwargs['temporal_encoder_nbr_hidden_units'])
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
        
        #TODO: find out whether use learning not target logit is anything interesting or not...:
        '''
        if not(self.use_learning_not_target_logit):
            l_shape = decision_logits.size()
            not_target_logit = torch.zeros( *l_shape[:2], 1)
        else:
            not_target_logit = self.not_target_logits_per_token.view((1,self.max_sentence_length,1)).repeat(batch_size, 1, 1)
        if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
        decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
        '''

        return decision_logits, self.embedding_tf_final_outputs


    def _utter(self, features, sentences):
        """
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) *temporal_feature_dim)`.
        """
        batch_size = features.size(0)
        nbr_distractors_po = features.size(1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, cnn_encoder_feature_dim)
        
        # Forward pass:
        if self.embedding_tf_final_outputs is None or self.embedding_tf_final_outputs.shape[1] != nbr_distractors_po: 
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
                # (batch_size*(nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
                outputs = outputs.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['temporal_encoder_nbr_hidden_units'])
                # (batch_size, (nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
                
                # Taking only the target features: assumes partial observations anyway...
                # TODO: find a way to compute the sentence while attending other features in case of full observability...
                embedding_tf_final_outputs = outputs[:,0,-1,:].contiguous()
                # (batch_size, kwargs['temporal_encoder_feature_dim'])
                self.embedding_tf_final_outputs = self.normalization(embedding_tf_final_outputs.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
                self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape(batch_size, nbr_distractors_po, -1)
                # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])
            else:
                self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
                self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, nbr_distractors_po, -1))
                # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])

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

        self.allowed_vocab_size = self.vocab_size//2
        if False:#self.train:
            logits = 0.5*torch.ones(self.vocab_size-1).float()
            logits[self.allowed_vocab_size] = 1.0
            # [0, ..., self.vocab_size-2]
            # allowed_vocab_size >= 2: 
            allowed_vocab_size = self.vocab_size - torch.distributions.categorical.Categorical(logits=logits).sample()
        else:
            allowed_vocab_size = self.vocab_size

        # Utter the next sentences:
        next_sentences_hidden_states, \
        next_sentences_widx, \
        next_sentences_logits, \
        next_sentences_one_hots = DifferentiableObverterAgent._compute_sentence(
            features_embedding=self.embedding_tf_final_outputs,
            target_idx=predicted_target_idx,
            symbol_encoder=self.symbol_encoder,
            symbol_processing=self.symbol_processing,
            symbol_decoder=self.symbol_decoder,
            init_rnn_states=self.rnn_states,
            allowed_vocab_size=allowed_vocab_size,
            vocab_size=self.vocab_size,
            max_sentence_length=self.max_sentence_length,
            nbr_distractors_po=nbr_distractors_po,
            operation=operation,
            vocab_stop_idx=self.vocab_size-1,
            use_obverter_threshold_to_stop_message_generation=self.kwargs['use_obverter_threshold_to_stop_message_generation'],
            use_stop_word=False,
            _compute_tau=self._compute_tau,
            not_target_logits_per_token=self.not_target_logits_per_token if self.use_learning_not_target_logit else None,
            use_sentences_one_hot_vectors=self.use_sentences_one_hot_vectors,
            logger=self.logger)

        return next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences_one_hots, self.embedding_tf_final_outputs

    def _compute_sentence(features_embedding, 
                          target_idx, 
                          symbol_encoder, 
                          symbol_processing, 
                          symbol_decoder, 
                          init_rnn_states=None,
                          allowed_vocab_size=10, 
                          vocab_size=10, 
                          max_sentence_length=14,
                          nbr_distractors_po=1,
                          operation=torch.max,
                          vocab_stop_idx=0,
                          use_obverter_threshold_to_stop_message_generation=False,
                          use_stop_word=False,
                          _compute_tau=None,
                          not_target_logits_per_token=None,
                          use_sentences_one_hot_vectors=False,
                          logger=None):
        """Compute sentences using the obverter approach, adapted to referential game variants following the
        descriptive approach described in the work of [Choi et al., 2018](http://arxiv.org/abs/1804.02341).

        In descriptive mode, `nbr_distractors_po=1` and `target_idx=torch.zeros((batch_size,1))`, 
        thus the algorithm behaves exactly like in Choi et al. (2018).
        Otherwise, the the likelyhoods for the target experience of being chosen by the decision module 
        is considered solely and the algorithm aims at maximizing/minimizing (following :param operation:) 
        this likelyhood over the sentence's next word.
        
        :param features_embedding: Tensor of (temporal) features embedding of shape `(batch_size, *self.obs_shape)`.
        :param target_idx: Tensor of indices of the target experiences of shape `(batch_size, 1)`.
        :param symbol_encoder: torch.nn.Module used to embed vocabulary indices into vocabulary embeddings.
        :param symbol_processing: torch.nn.Module used to generate the sentences.
        :param symbol_decoder: torch.nn.Module used to decode the embeddings generated by the `:param symbol_processing:` module. 
        :param init_rnn_states: None or Tuple of Tensors to initialize the symbol_processing's rnn states.
        :param vocab_size: int, size of the vocabulary.
        :param max_sentence_length: int, maximal length for each generated sentences.
        :param nbr_distractors_po: int, number of distractors and target, i.e. `nbr_distractors+1.
        :param operation: Function, expect `torch.max` or `torch.min`.
        :param vocab_stop_idx: int, index of the STOP symbol in the vocabulary.
        :param use_obverter_threshold_to_stop_message_generation:  boolean, or float that specifies whether to stop the 
                                                                    message generation when the decision module's 
                                                                    output probability is abobe a given threshold 
                                                                    (or below it if the operation is `torch.min`).
                                                                    If it is a float, then it is the value of the threshold.
        :param use_stop_word: boolean that specifies whether to use one of the word in the vocabulary with a pre-defined meaning,
                              that is that it is a STOP token, thus effictively ending the symbol generation for the current sentence.
        
        :returns:
            - sentences_widx: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], 1)` where `b` is the batch index.
                             It represents the indices of the chosen words.
            - sentences_logits: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the logits of words over the decision module's potential to choose the target experience as output.
            - sentences_one_hots: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the sentences as one-hot-encoded word vectors.
        
        """
        batch_size = features_embedding.size(0)
        states = init_rnn_states
        
        arange_vocab = torch.arange(vocab_size).float()
        arange_allowed_vocab = torch.arange(allowed_vocab_size).float()
        if features_embedding.is_cuda: 
            arange_vocab = arange_vocab.cuda()
            arange_allowed_vocab = arange_allowed_vocab.cuda()
        
        if use_sentences_one_hot_vectors:
            vocab_idx = torch.zeros((allowed_vocab_size, vocab_size))
            # (allowed_vocab_size, vocab_size)
            for i in range(allowed_vocab_size): vocab_idx[i,i] = 1.0
        else:
            vocab_idx = torch.zeros((allowed_vocab_size,1)).long()
            # (allowed_vocab, 1)
            for i in range(allowed_vocab_size): vocab_idx[i] = i
        if features_embedding.is_cuda: vocab_idx = vocab_idx.cuda()
        vocab_idx = symbol_encoder(vocab_idx).view((allowed_vocab_size,1,-1))
        # Embedding: (batch_size=allowed_vocab_size, 1, kwargs['symbol_embedding_size'])

        sentences_hidden_states = [list() for _ in range(batch_size)]
        sentences_widx = [list() for _ in range(batch_size)]
        sentences_logits = [list() for _ in range(batch_size)]
        sentences_one_hots = [list() for _ in range(batch_size)]
        for b in range(batch_size):
            bemb = features_embedding[b].view((nbr_distractors_po, -1))
            # ( (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
            btarget_idx = target_idx[b]
            # (1,)
            continuer = True
            sentence_token_count = 0
            while continuer:
                sentence_token_count += 1
                if states is not None:
                    '''
                    hs, cs = states[0], states[1]
                    hs = hs.repeat( 1, vocab_size, 1)
                    cs = cs.repeat( 1, vocab_size, 1)
                    rnn_states = (hs, cs)
                    '''
                    rnn_states = states.repeat(1,allowed_vocab_size, 1)
                else :
                    rnn_states = states

                rnn_outputs, next_rnn_states = symbol_processing(vocab_idx, rnn_states )
                # (batch_size=allowed_vocab_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
                # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
                
                # Compute the decision: following the last hidden/output vector from the rnn:
                decision_inputs = rnn_outputs[:,-1,...]
                # (batch_size=allowed_vocab_size, kwargs['symbol_processing_nbr_hidden_units'])
                decision_logits = []
                for bv in range(allowed_vocab_size):
                    bdin = decision_inputs[bv].unsqueeze(1)
                    # (kwargs['symbol_processing_nbr_hidden_units'], 1)
                    dl = torch.matmul( bemb, bdin).view((1,-1))
                    # ( 1, (nbr_distractors+1))
                    decision_logits.append(dl)
                decision_logits = torch.cat(decision_logits, dim=0)
                # (batch_size=allowed_vocab_size, (nbr_distractors+1) )
                
                if not_target_logits_per_token is None:
                    not_target_logit = torch.zeros(decision_logits.size(0), 1)
                else:
                    not_target_logit = not_target_logits_per_token[:,sentence_token_count-1].repeat(allowed_vocab_size,1)
                if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
                decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
                # (batch_size=allowed_vocab_size, (nbr_distractors+2) )
                # If the game is a descriptive game, then there is a possibility that none of the stimulus are target.
                # This added element accounts for it.
                
                tau0 = 1e1
                # The higher this value is, the sharper the computed categorical distribution is.
                # Probs over Distractors and Vocab: 
                decision_probs = F.softmax( decision_logits.view(-1), dim=-1).view((allowed_vocab_size, -1))
                decision_probs_least_effort = F.softmax( decision_logits.view(-1)*tau0, dim=-1).view((allowed_vocab_size, -1))
                # (batch_size=vocab_size, (nbr_distractors+2) )
                
                target_decision_probs_per_vocab_logits = decision_probs[:,btarget_idx]
                target_decision_probs_least_effort_per_vocab_logits = decision_probs_least_effort[:,btarget_idx]
                # (batch_size=allowed_vocab_size, )
                # TODO: it might be relevant to treat those values as logits values for the next sampling of the actual token?

                tau = 1.0/5e0 
                if _compute_tau is not None:    tau = _compute_tau(tau0=tau, h=rnn_outputs[btarget_idx])
                if logger is not None: 
                    it = 0
                    key = "Obverter/ComputeSentenceTau"
                    logger.add_scalar(key, tau.item(), it)

                tau = tau.view((-1))
                tau1 = 5e1
                # The closer to zero this value is, the more accurate the operation is.
                straight_through = True
                one_hot_sampled_vocab = gumbel_softmax(logits=target_decision_probs_per_vocab_logits*tau1, tau=tau, hard=straight_through, dim=-1)
                # (batch_size=allowed_vocab_size,)

                if allowed_vocab_size < vocab_size:
                    zeros4complete_vocab = torch.zeros((vocab_size-allowed_vocab_size,))
                    if one_hot_sampled_vocab.is_cuda: zeros4complete_vocab = zeros4complete_vocab.cuda()
                    one_hot_sampled_vocab = torch.cat([one_hot_sampled_vocab, zeros4complete_vocab], dim=0)
                    
                    target_decision_probs_per_vocab_logits = torch.cat([target_decision_probs_per_vocab_logits,
                                                                                     zeros4complete_vocab], dim=0)
                    target_decision_probs_least_effort_per_vocab_logits = torch.cat([target_decision_probs_least_effort_per_vocab_logits,
                                                                                     zeros4complete_vocab], dim=0)
                
                vocab_idx_argop = torch.sum(arange_vocab*one_hot_sampled_vocab)
                vocab_idx_op = target_decision_probs_per_vocab_logits[vocab_idx_argop.long()]
                # Or make it easier for the sentence to be stopped by looking at the high-temperature softmax distribution:
                # The higher the temperature, the larger the differences between each categories probability. 
                #vocab_idx_op = target_decision_probs_least_effort_per_vocab_logits[vocab_idx_argop.long()]
                
                sentences_hidden_states[b].append(rnn_outputs.view(1,-1))
                sentences_widx[b].append( vocab_idx_argop)
                sentences_logits[b].append( target_decision_probs_least_effort_per_vocab_logits.view((1,-1)))
                sentences_one_hots[b].append( nn.functional.one_hot(vocab_idx_argop.long(), num_classes=vocab_size).view((1,-1)))
                
                # next rnn_states:
                #states = [st[-1, vocab_idx_argop].view((1,1,-1)) for st in next_rnn_states]
                states = next_rnn_states[-1, vocab_idx_argop.long()].view((1,1,-1))

                if use_obverter_threshold_to_stop_message_generation:
                    if operation == torch.max:
                        operation_condition = (vocab_idx_op >= use_obverter_threshold_to_stop_message_generation)
                    else:
                        operation_condition = (vocab_idx_op < 1-use_obverter_threshold_to_stop_message_generation) 
                else:
                    operation_condition = False
                
                if use_stop_word:
                    stop_word_condition = (vocab_idx_argop.long() == vocab_stop_idx)
                else:
                    stop_word_condition = False 

                if len(sentences_widx[b]) >= max_sentence_length or stop_word_condition or operation_condition:
                    continuer = False

            # Embed the sentence:

            # Padding token:
            while len(sentences_widx[b]) < max_sentence_length:
                # Padding with PAD token index. Which implies that padding is part of the vocabulary, weird right?
                # It is okay, this embedding is never actually used or made relevant since the decoding only goes
                # until vocab_size-1 index (whereas PAD is index vocab_size), and the RNN states is truncated
                # on the values that are actually proper tokens. It is not the last rnn_state that is to be used
                # in later computations.
                sentences_widx[b].append((vocab_size)*torch.ones_like(vocab_idx_argop))

            sentences_hidden_states[b] = torch.cat(sentences_hidden_states[b], dim=0)
            # (sentence_length<=max_sentence_length, kwargs['symbol_preprocessing_nbr_hidden_units'])
            sentences_widx[b] = torch.cat([ word_idx.view((1,1,-1)) for word_idx in sentences_widx[b]], dim=1)
            # (batch_size=1, sentence_length<=max_sentence_length, 1)
            sentences_logits[b] = torch.cat(sentences_logits[b], dim=0)
            # (sentence_length<=max_sentence_length, vocab_size)
            sentences_one_hots[b] = torch.cat(sentences_one_hots[b], dim=0) 
            # (sentence_length<=max_sentence_length, vocab_size)

            # Reset the state for the next sentence generation in the batch:
            states = init_rnn_states

        sentences_one_hots = nn.utils.rnn.pad_sequence(sentences_one_hots, batch_first=True, padding_value=0.0).float()
        # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)
        
        sentences_widx = torch.cat(sentences_widx, dim=0)
        # (batch_size, max_sentence_length, 1)
        if features_embedding.is_cuda: sentences_widx = sentences_widx.cuda()

        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots
