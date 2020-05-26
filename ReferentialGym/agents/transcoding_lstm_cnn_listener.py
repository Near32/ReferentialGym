import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminative_listener import DiscriminativeListener
from ..networks import choose_architecture, layer_init, hasnan, BetaVAE
from ..utils import StraightThroughGumbelSoftmaxLayer


class TranscodingLSTMCNNListener(DiscriminativeListener):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='l0', logger=None):
        """
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        """
        super(TranscodingLSTMCNNListener, self).__init__(obs_shape, vocab_size, max_sentence_length, agent_id, logger, kwargs)
        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        MHDPANbrHead=4
        MHDPANbrRecUpdate=1
        MHDPANbrMLPUnit=512
        MHDPAInteractionDim=128
        if 'mhdpa_nbr_head' in self.kwargs: MHDPANbrHead = self.kwargs['mhdpa_nbr_head']
        if 'mhdpa_nbr_rec_update' in self.kwargs: MHDPANbrRecUpdate = self.kwargs['mhdpa_nbr_rec_update']
        if 'mhdpa_nbr_mlp_unit' in self.kwargs: MHDPANbrMLPUnit = self.kwargs['mhdpa_nbr_mlp_unit']
        if 'mhdpa_interaction_dim' in self.kwargs: MHDPAInteractionDim = self.kwargs['mhdpa_interaction_dim']

        if 'cnn_encoder' in self.kwargs:
            self.cnn_encoder = self.kwargs['cnn_encoder']
        else:
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
            
            if not('cnn_encoder' in self.kwargs):
                self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                       kwargs=self.kwargs,
                                                       input_shape=cnn_input_shape,
                                                       feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                       dropout=self.kwargs['dropout_prob'])
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
            self.temporal_feature_encoder = layer_init(nn.LSTM(input_size=temporal_encoder_input_dim,
                                              hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                              num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                              batch_first=True,
                                              dropout=self.kwargs['dropout_prob'],
                                              bidirectional=False))
        else:
            self.temporal_feature_encoder = None
            print("WARNING: Symbol processing :: the number of hidden units is being reparameterized to fit to convolutional features.")
            self.kwargs['temporal_encoder_nbr_hidden_units'] = self.kwargs['nbr_stimulus']*self.encoder_feature_shape
            #self.kwargs['symbol_processing_nbr_hidden_units'] = self.kwargs['temporal_encoder_nbr_hidden_units']


        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['temporal_encoder_nbr_hidden_units'])
        #self.normalization = nn.LayerNorm(normalized_shape=self.kwargs['temporal_encoder_nbr_hidden_units'])
        
        ## Textual Encoder:        
        self.textual_embedding_size = self.kwargs['symbol_embedding_size']
        self.textual_embedder = nn.Sequential(
            nn.Linear(self.vocab_size, self.textual_embedding_size, bias=False),
            nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
            )

        self.textual_encoder_input_dim = self.textual_embedding_size
        self.textual_encoder_hidden_size = self.kwargs['textual_encoder_nbr_hidden_units']
        self.textual_encoder = nn.LSTM(input_size=self.textual_encoder_input_dim,
                                      hidden_size=self.textual_encoder_hidden_size, 
                                      num_layers=self.kwargs['textual_encoder_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)
        self.textual_encoder_learnable_initial_state = nn.Parameter(
            torch.zeros(1,1,self.textual_encoder_hidden_size)
        )
        

        ## Multi-modal Transcoder:
        self.decoder_nbr_steps = self.kwargs['visual_decoder_nbr_steps']
        # This transcoder has to embed the decoder hidden state.
        # Therefore, the embedding can be a simple MLP:
        self.visual_decoder_hidden_size = self.kwargs['visual_decoder_nbr_hidden_units']
        self.transcoder_visual_embedder_input_size = self.visual_decoder_hidden_size
        self.transcoder_visual_embedder_output_size = self.encoder_feature_shape
        self.transcoder_visual_embedder =  nn.Linear(
            self.transcoder_visual_embedder_input_size, 
            self.transcoder_visual_embedder_output_size, 
            bias=False
        )
        self.transcoder_visual_embedder_learnable_initial_input = nn.Parameter(
            torch.zeros(1,self.transcoder_visual_embedder_input_size)
        )
        
        # 
        self.transcoder_hidden_size = self.kwargs['transcoder_nbr_hidden_units']
        self.transcoder_input_size = self.transcoder_visual_embedder_output_size
        self.transcoder = nn.LSTM(input_size=self.transcoder_input_size,
          hidden_size=self.transcoder_hidden_size, 
          num_layers=self.kwargs['transcoder_nbr_rnn_layers'],
          batch_first=True,
          dropout=self.kwargs['dropout_prob'],
          bidirectional=False,
        )
        self.transcoder_learnable_initial_state = nn.Parameter(
            torch.zeros(1,1,self.transcoder_hidden_size)
        )
        
        # MLP + Concat Attention:
        self.transcoder_attention_interaction_dim = self.kwargs['transcoder_attention_interaction_dim']
        self.transcoder_guided_textual_pre_att = nn.Sequential(
            nn.Linear(
                self.transcoder_hidden_size+self.textual_encoder_hidden_size, 
                self.transcoder_attention_interaction_dim, 
                bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.transcoder_guided_textual_att = nn.Linear(self.transcoder_attention_interaction_dim, 1, bias=False)

        self.textual_context_dim = self.textual_embedding_size

        self.st_gs = StraightThroughGumbelSoftmaxLayer(
                inv_tau0=self.kwargs['transcoder_st_gs_inv_tau0'], 
                input_dim=self.transcoder_hidden_size
        )

        ##

        ## Visual Decoder:
        self.visual_decoder_input_dim = self.textual_context_dim+self.transcoder_visual_embedder_output_size
        self.visual_decoder = nn.LSTM(input_size=self.visual_decoder_input_dim,
                                      hidden_size=self.visual_decoder_hidden_size, 
                                      num_layers=self.kwargs['visual_decoder_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)
        self.visual_decoder_learnable_initial_state = nn.Parameter(
                torch.zeros(1,1,self.visual_decoder_hidden_size)
        )

        self.visual_decoder_mlp = nn.Sequential(
            nn.Linear(
                self.visual_decoder_hidden_size, 
                self.encoder_feature_shape, 
                bias=True
            ),
            nn.Dropout( p=self.kwargs['visual_decoder_mlp_dropout_prob'])
        )

        if self.textual_context_dim != self.visual_decoder_hidden_size:
            self.textual_context2decoder_converter = nn.Linear(
                self.textual_context_dim,
                self.visual_decoder_hidden_size
            )
        else:
            self.textual_context2decoder_converter = None 

        ##
        
        '''
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, self.kwargs['max_sentence_length'], 1)))
        '''

        self.projection_normalization = None #nn.BatchNorm1d(num_features=self.kwargs['max_sentence_length']*self.kwargs['symbol_processing_nbr_hidden_units'])

        self.reset()

    def reset(self):
        self.textual_embedder.apply(layer_init)
        self.textual_encoder.apply(layer_init)
        self.transcoder_visual_embedder.apply(layer_init)
        self.transcoder.apply(layer_init)
        self.transcoder_guided_textual_pre_att.apply(layer_init)
        self.transcoder_guided_textual_att.apply(layer_init)
        self.st_gs.apply(layer_init)
        self.visual_decoder.apply(layer_init)
        self.visual_decoder_mlp.apply(layer_init)
        if self.textual_context2decoder_converter is not None:
            self.textual_context2decoder_converter.apply(layer_init)
        
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        self.embedding_tf_final_outputs = None

        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

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
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
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
            else:
                featout = self.cnn_encoder(stin)
                if self.use_feat_converter:
                    featout = self.featout_converter(featout)

            features.append(featout)
        
        self.features = self.cnn_encoder_normalization(torch.cat(features, dim=0))
        
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

        # (Eq 1) :
        # (batch_size, max_sentence_length, self.vocab_size)
        sentences_length = sentences.shape[1]
        sentences = sentences.view((-1, self.vocab_size))
        embedded_symbols = self.textual_embedder(sentences) 
        # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
        embedded_sentences = embedded_symbols.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
        # (batch_size, max_sentence_length, textual_embedding_size)
        
        # (Eq 2) :
        init_textual_encoder_state = self.textual_encoder_learnable_initial_state.expand(
            self.kwargs['textual_encoder_nbr_rnn_layers'],
            batch_size, 
            -1
        )
        # (batch_size=1, nbr_hidden_layers*num_directions=1, self.textual_encoder_hidden_size)
        textual_encoder_state = (
                init_textual_encoder_state.contiguous(), 
                torch.zeros_like(init_textual_encoder_state)
            )
        textual_encoder_outputs, next_textual_encoder_states = self.textual_encoder(
            embedded_sentences, 
            textual_encoder_state
        )
        # (batch_size, sentences_length, self.textual_encoder_hidden_size)
        # (hidden_layer*num_directions, batch_size, self.textual_encoder_hidden_size)
        
        # Batch Normalization:
        if self.projection_normalization is not None:
            textual_encoder_outputs = self.projection_normalization(textual_encoder_outputs.reshape((batch_size, -1)))
            textual_encoder_outputs = textual_encoder_outputs.reshape((batch_size, -1, self.textual_encoder_hidden_size))

        #  Transcoder+Decoder Loop Initialization:
        init_transcoder_state = self.transcoder_learnable_initial_state.expand(
            self.kwargs['transcoder_nbr_rnn_layers'],
            batch_size, 
            -1
        )
        # (hidden_layer*num_directions=1, batch_size, kwargs['transcoder_nbr_hidden_units'])
        transcoder_state = (
            init_transcoder_state.contiguous(),
            torch.zeros_like(init_transcoder_state)
        )

        # Decoder initial state, as seen in (Eq 10), after full focus, if any...
        init_visual_decoder_state = self.visual_decoder_learnable_initial_state.expand(
            self.kwargs['transcoder_nbr_rnn_layers'],
            batch_size, 
            -1
        )
        # (hidden_layer*num_directions=1, batch_size=1, self.symbol_processing_hidden_size)
        visual_decoder_state = (
            init_visual_decoder_state.contiguous(),
            torch.zeros_like(init_visual_decoder_state)
        )

        visual_decoder_output = self.transcoder_visual_embedder_learnable_initial_input.expand(
            batch_size, 
            -1
        )

        decision_logits = []
        per_step_visual_decoder_mlp_outputs = []
        for timestep in range(self.decoder_nbr_steps):
            ## Transcoder:
            # (Eq 3) Encoding the decoder output symbol at t-1:
            transcoder_visual_embedding = self.transcoder_visual_embedder(visual_decoder_output).unsqueeze(1)
            # (batch_size, 1, transcoder_visual_embedder_output_size)
            # (Eq 4) Forward pass through the transcoder:
            transcoder_outputs, next_transcoder_states = self.transcoder(transcoder_visual_embedding, transcoder_state)
            # (batch_size, 1, kwargs['transcoder_nbr_hidden_units'])
            # (hidden_layer*num_directions, batch_size, kwargs['transcoder_nbr_hidden_units'])
            # (Eq 5): assuming LSTMs... taking the last layer output:
            transcoder_state = transcoder_state[0][-1].unsqueeze(1).expand(-1, sentences_length,-1)
            # (batch_size, sentences_length, transcoder_nbr_hidden_size)
            trans_att_inputs = torch.cat([textual_encoder_outputs, transcoder_state], dim=-1)
            # (batch_size, sentence_length, textual_encoder_hidden_size+transcoder_hidden_size) 
            # (Eq 5): ReLU Linear
            trans_pre_att_activation = self.transcoder_guided_textual_pre_att(trans_att_inputs)
            # ( batch_size, sentence_length, interaction_dim) 
            # (Eq 5 & 6): Final Linear + Softmax
            trans_att = self.transcoder_guided_textual_att(trans_pre_att_activation).reshape(batch_size, sentences_length).softmax(dim=-1)
            # ( batch_size, sentence_length)
            
            # (Eq 7 & 8) Straight-Through Gumbel-Softmax:
            if not self.kwargs['transcoder_listener_soft_attention']:
                trans_att = self.st_gs(logits=trans_att, param=transcoder_state)
            # (batch_size, sentence_length)
            # (Eq 9) Context computation:
            context = (trans_att.unsqueeze(-1) * embedded_sentences).sum(1) 
            # (batch_size, context_dim=textual_embedding_size)

            ## Decoder:
            # (Eq 10) Forward pass through the decoder:
            visual_decoder_input = torch.cat([context.unsqueeze(1), transcoder_visual_embedding], dim=-1)
            # (batch_size, seq_len=1, textual_embedding_size+transcoder_visual_embedder_output_size)
            visual_decoder_outputs, next_visual_decoder_states = self.visual_decoder(
                visual_decoder_input, 
                visual_decoder_state
            )
            # (batch_size, 1, visual_decoder_hidden_size)
            # (hidden_layer*num_directions, batch_size, visual_decoder_hidden_size)
            # (Eq 11 - listener)
            visual_decoder_outputs = visual_decoder_outputs.reshape(batch_size,-1)
            # (batch_size, visual_decoder_hidden_size)
            visual_decoder_mlp_output = self.visual_decoder_mlp(visual_decoder_outputs)
            # (batch_size, self.encoded_feature_shape)
            per_step_visual_decoder_mlp_outputs.append(visual_decoder_mlp_output)

            ## Bookkeeping:
            # Transcoder
            visual_decoder_output = visual_decoder_outputs
            transcoder_state = next_transcoder_states
            # Visual Decoder:
            visual_decoder_state = next_visual_decoder_states
            # Visual Decoder's Full Focus Scheme:
            if self.textual_context2decoder_converter is not None:
                context = self.textual_context2decoder_converter(context.reshape(-1, self.textual_context_dim))
            context = context.reshape(batch_size,1,-1)
            # (batch_size, 1, visual_decoder_hidden_size)
            # Assumes LSTMs...
            full_focus_visual_decoder_state = visual_decoder_state[0]*context
            decoder_state =(
                full_focus_visual_decoder_state,
                visual_decoder_state[-1]
            )

            ## Visual Feature Projection:
            stimuli_features = self.embedding_tf_final_outputs
            # (batch_size, (nbr_distractors+1), encoded_feature_shape)
            nbr_distractors_po = self.embedding_tf_final_outputs.shape[1]
            # Summing on all the previous visual decoder output:
            visually_projected_textual_features = torch.stack(
                per_step_visual_decoder_mlp_outputs).sum(dim=0).unsqueeze(1
            ).expand(
                -1,
                nbr_distractors_po,
                -1
            )
            # (batch_size, (nbr_distractros+1), self.encoded_feature_shape)
            decision_logits_until_timestep = (visually_projected_textual_features*stimuli_features).sum(-1)
            # ( batch_size, (nbr_distractors+1))
            decision_logits.append(decision_logits_until_timestep.unsqueeze(1))
            # (batch_size, 1, (nbr_distractors+1) )
        decision_logits = torch.cat(decision_logits, dim=1)
        # (batch_size, nbr_decoder_steps, (nbr_distractors+1))           

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
