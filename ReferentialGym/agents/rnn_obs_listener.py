import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminative_listener import DiscriminativeListener
from .rnn_cnn_listener import RNNCNNListener
from ..networks import choose_architecture, layer_init, hasnan, BetaVAE


class RNNObsListener(RNNCNNListener):
    def __init__(
        self,
        kwargs, 
        obs_shape, 
        vocab_size=100, 
        max_sentence_length=10, 
        agent_id='l0', 
        logger=None,
        rnn_type='lstm',
    ):
        """
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param rnn_type: String specifying the type of RNN to use, either 'gru' or 'lstm'.
        """
        DiscriminativeListener.__init__(
            self,
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs,
        )
        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 

        MHDPANbrHead=4
        MHDPANbrRecUpdate=1
        MHDPANbrMLPUnit=512
        MHDPAInteractionDim=128
        if 'mhdpa_nbr_head' in self.kwargs: MHDPANbrHead = self.kwargs['mhdpa_nbr_head']
        if 'mhdpa_nbr_rec_update' in self.kwargs: MHDPANbrRecUpdate = self.kwargs['mhdpa_nbr_rec_update']
        if 'mhdpa_nbr_mlp_unit' in self.kwargs: MHDPANbrMLPUnit = self.kwargs['mhdpa_nbr_mlp_unit']
        if 'mhdpa_interaction_dim' in self.kwargs: MHDPAInteractionDim = self.kwargs['mhdpa_interaction_dim']

        if 'obs_encoder' in self.kwargs:
            self.obs_encoder = self.kwargs['obs_encoder']
        elif len(self.obs_shape)==3:
            input_shape = self.obs_shape[2:]
            self.obs_encoder = choose_architecture(
                architecture=self.kwargs['architecture'],
                kwargs=self.kwargs,
                input_shape=input_shape,
                fc_hidden_units_list=self.kwargs['fc_hidden_units'],
                dropout=self.kwargs['dropout_prob'],
            )
        elif len(self.obs_shape)==5:
            cnn_input_shape = self.obs_shape[2:]
            self.obs_encoder = choose_architecture(
                architecture=self.kwargs['architecture'],
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
                MHDPAInteractionDim=MHDPAInteractionDim,
            )
        else:
            raise NotImplementedError
        
        self.use_feat_converter = self.kwargs['use_feat_converter'] if 'use_feat_converter' in self.kwargs else False 
        if self.use_feat_converter:
            self.feat_converter_input = self.obs_encoder.get_feature_shape()


        if 'BetaVAE' in self.kwargs['architecture'] or 'MONet' in self.kwargs['architecture']:
            self.VAE_losses = list()
            self.compactness_losses = list()
            self.buffer_cnn_output_dict = dict()
            
            if not('obs_encoder' in self.kwargs):
                self.obs_encoder = choose_architecture(
                    architecture=self.kwargs['architecture'],
                    kwargs=self.kwargs,
                    input_shape=cnn_input_shape,
                    feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                    dropout=self.kwargs['dropout_prob'],
                )
            # N.B: with a VAE, we want to learn the weights in any case:
            if 'agent_learning' in self.kwargs:
                assert('transfer_learning' not in self.kwargs['agent_learning'])
            
            self.vae_detached_featout = False
            if self.kwargs['vae_detached_featout']:
                self.vae_detached_featout = True

            self.VAE = self.obs_encoder

            self.use_feat_converter = True
            self.feat_converter_input = self.obs_encoder.latent_dim
        else:
            if 'agent_learning' in self.kwargs and 'transfer_learning' in self.kwargs['agent_learning']:
                self.obs_encoder.detach_conv_maps = True

        self.encoder_feature_shape = self.obs_encoder.get_feature_shape()
        if self.use_feat_converter:
            self.featout_converter = []
            hidden = int((self.feat_converter_input+self.kwargs['feat_converter_output_size'])/2)
            self.featout_converter.append(nn.Linear(self.feat_converter_input, hidden))
            self.featout_converter.append(nn.BatchNorm1d(num_features=hidden))
            self.featout_converter.append(nn.ReLU())
            self.featout_converter.append(nn.Linear(hidden, self.kwargs['feat_converter_output_size'])) 
            self.featout_converter.append(nn.BatchNorm1d(num_features=self.kwargs['feat_converter_output_size']))
            self.featout_converter.append(nn.ReLU())
            self.featout_converter =  nn.Sequential(*self.featout_converter)
            self.encoder_feature_shape = self.kwargs['feat_converter_output_size']
        
        self.obs_encoder_normalization = nn.BatchNorm1d(num_features=self.encoder_feature_shape)
        
        temporal_encoder_input_dim = self.obs_encoder.get_feature_shape()
        if self.kwargs['temporal_encoder_nbr_rnn_layers'] > 0:
            self.temporal_feature_encoder = layer_init(
                nn.LSTM(input_size=temporal_encoder_input_dim,
                hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                batch_first=True,
                dropout=self.kwargs['dropout_prob'],
                bidirectional=False,
            ))
        else:
            self.temporal_feature_encoder = None
            print("WARNING: Symbol processing :: the number of hidden units is being reparameterized to fit to convolutional features.")
            self.kwargs['temporal_encoder_nbr_hidden_units'] = self.kwargs['nbr_stimulus']*self.encoder_feature_shape
            self.kwargs['symbol_processing_nbr_hidden_units'] = self.kwargs['temporal_encoder_nbr_hidden_units']

        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['temporal_encoder_nbr_hidden_units'])
        #self.normalization = nn.LayerNorm(normalized_shape=self.kwargs['temporal_encoder_nbr_hidden_units'])
    
        symbol_processing_input_dim = self.kwargs['symbol_embedding_size']
        self.rnn_type = rnn_type.lower()
        if 'lstm' in self.rnn_type:
            self.symbol_processing = nn.LSTM(
                input_size=symbol_processing_input_dim,
                hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                batch_first=True,
                dropout=self.kwargs['dropout_prob'],
                bidirectional=False,
            )
        elif 'gru' in self.rnn_type:
            self.symbol_processing = nn.GRU(
                input_size=symbol_processing_input_dim,
                hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                batch_first=True,
                dropout=self.kwargs['dropout_prob'],
                bidirectional=False,
            )
        else:
            raise NotImplementedError
        '''
        self.symbol_processing_learnable_initial_state = nn.Parameter(
                torch.zeros(1,1,self.kwargs['symbol_processing_nbr_hidden_units'])
        )
        '''
        
        self.symbol_encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False),
            nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
        )
        
        self.tau_fc = nn.Sequential(
            nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
            nn.Softplus(),
        )

        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, 1, 1)))
        
        self.projection_normalization = None #nn.BatchNorm1d(num_features=self.kwargs['max_sentence_length']*self.kwargs['symbol_processing_nbr_hidden_units'])

        self.reset()
        
        if self.kwargs['use_cuda']:
            self = self.cuda()

    def reset(self, reset_language_model=False, whole=False):
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()
        if whole:
            self.obs_encoder.apply(layer_init)

    def _tidyup(self):
        """
        Called at the agent level at the end of the `compute` function.
        """
        self.embedding_tf_final_outputs = None

        if isinstance(self.obs_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

    
    def _sense(self, experiences, sentences=None, **kwargs):
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
        mini_batch_size = min(self.kwargs['mini_batch_size'], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            if isinstance(self.obs_encoder, BetaVAE):
                cnn_output_dict  = self.obs_encoder.compute_loss(stin)
                if 'VAE_loss' in cnn_output_dict:
                    self.VAE_losses.append(cnn_output_dict['VAE_loss'])
                
                if hasattr(self.obs_encoder, 'compactness_losses') and self.obs_encoder.compactness_losses is not None:
                    self.compactness_losses.append(self.obs_encoder.compactness_losses.cpu())
                
                for key in cnn_output_dict:
                    if key not in self.buffer_cnn_output_dict:
                        self.buffer_cnn_output_dict[key] = list()
                    self.buffer_cnn_output_dict[key].append(cnn_output_dict[key].cpu())

                if self.kwargs['vae_use_mu_value']:
                    featout = self.obs_encoder.mu 
                else:
                    featout = self.obs_encoder.z

                if self.vae_detached_featout:
                    featout = featout.detach()

                featout = self.featout_converter(featout)
            else:
                featout = self.obs_encoder(stin)
                if self.use_feat_converter:
                    if len(featout.shape)>2:    
                        featout = featout.reshape(stin.shape[0], -1)
                    featout = self.featout_converter(featout)

            features.append(featout)
        
        self.features = torch.cat(features, dim=0).reshape((-1, featout.shape[-1]))
        self.features = self.obs_encoder_normalization(self.features)
        
        self.features = self.features.view(batch_size, nbr_distractors_po, self.config['nbr_stimulus'], -1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        if isinstance(self.obs_encoder, BetaVAE):
            self.VAE_losses = torch.cat(self.VAE_losses).contiguous()#.view((batch_size,-1)).mean(dim=-1)
            
            for key in self.buffer_cnn_output_dict:
                self.log_dict[key] = torch.cat(self.buffer_cnn_output_dict[key]).mean()

            self.log_dict['kl_capacity'] = torch.Tensor([100.0*self.obs_encoder.EncodingCapacity/self.obs_encoder.maxEncodingCapacity])
            if len(self.compactness_losses):
                self.log_dict['unsup_compactness_loss'] = torch.cat(self.compactness_losses).mean()

        return self.features 

     
