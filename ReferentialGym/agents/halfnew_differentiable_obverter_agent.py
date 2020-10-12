import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminative_listener import DiscriminativeListener
from ..networks import choose_architecture, layer_init, BetaVAE, reg_nan, hasnan
from ..utils import gumbel_softmax

use_decision_head = True
nbr_head_outputs = 2
not_always_argmax = False ; always_categorical_sampling = False

LogSoftmaxAfterLogOnSigmoid = False

bmm = False
normalize = True 

inner_model = True
use_one_minus_max_prob = True
whole_sentence = True

packpadding = False 
assume_padding_with_eos = True

stability_eps = 1e-8
scale_negative_logit = 0.0 #-1e4
use_stop_word_in_compute_sentence = False 


def sentence_length_entropy_logging_hook(agent,
                                 losses_dict,
                                 input_streams_dict,
                                 outputs_dict,
                                 logs_dict,
                                 **kwargs):
    it_rep = input_streams_dict['it_rep']
    it_comm_round = input_streams_dict['it_comm_round']
    mode = input_streams_dict['mode']
    config = input_streams_dict['config']

    if 'speaker' not in agent.role: return

    batch_size = len(input_streams_dict['experiences'])
    speaker_sentences_logits = outputs_dict["sentences_logits"]
    speaker_sentences_widx = outputs_dict["sentences_widx"]

    # Sentence Lengths:
    """
    sentence_lengths = torch.sum(-(speaker_sentences_widx.squeeze(-1)-agent.vocab_size).sign(), dim=-1).reshape(batch_size,-1)
    """
    eos_mask = (speaker_sentences_widx.squeeze(-1)==agent.vocab_stop_idx)
    padding_with_eos = assume_padding_with_eos #(eos_mask.cumsum(-1).sum()>batch_size)
    # Include first EoS Symbol:
    if padding_with_eos:
        token_mask = ((eos_mask.cumsum(-1)>1)<=0)
        lengths = token_mask.sum(-1)
        #(batch_size, )
    else:
        token_mask = ((eos_mask.cumsum(-1)>0)<=0)
        lengths = token_mask.sum(-1)
        
    if not(padding_with_eos):
        # If excluding first EoS:
        lengths = lengths.add(1)
    sentence_lengths = lengths.clamp(max=agent.max_sentence_length).float()
    #(batch_size, )
    
    sentence_length = sentence_lengths.mean()
    
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{agent.agent_id}/SentenceLength (/{config['max_sentence_length']})"] = sentence_lengths/config['max_sentence_length']

    # Compute Sentence Entropies:
    # Assert that it is a probability distribution by applying softmax... 
    # (like probs values would be computed from logits in Categorical...)
    
    """
    sentences_log_probs = [
        #s_logits.reshape(-1, agent.vocab_size).log_softmax(dim=-1)#*sentence_mask.float().reshape(-1, 1)
        #s_logits.reshape(-1, agent.vocab_size)*sentence_mask.float().reshape(-1, 1)
        s_logits.reshape(-1, agent.vocab_size) #*sentence_mask.float().reshape(-1, 1)
        for s_logits, sentence_mask in zip(speaker_sentences_logits, token_mask)
    ]
    """
    
    sentences_log_probs = [
        s_logits.reshape(-1, agent.vocab_size) 
        for s_logits in speaker_sentences_logits
    ]
    
    speaker_sentences_log_probs = torch.cat(
        [ s_log_probs.gather(dim=-1,index=s_widx[:s_log_probs.shape[0]].long()).sum().unsqueeze(0) 
          for s_log_probs, s_widx in zip(sentences_log_probs, speaker_sentences_widx)
        ], 
        dim=0
    )
    
    entropies_per_sentence = -(speaker_sentences_log_probs.exp() * speaker_sentences_log_probs)
    # (batch_size, )
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{agent.agent_id}/Entropy"] = entropies_per_sentence.mean().item()

    perplexities_per_sentence = speaker_sentences_log_probs.exp().pow(1.0/sentence_lengths)
    # (batch_size, )
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{agent.agent_id}/SentenceLengthNormalizedPerplexity"] = perplexities_per_sentence.mean().item()


class DifferentiableObverterAgent(DiscriminativeListener):
    def __init__(self,
                 kwargs, 
                 obs_shape, 
                 vocab_size=100, 
                 max_sentence_length=10, 
                 agent_id='o0', 
                 logger=None, 
                 use_sentences_one_hot_vectors=True,
                 use_decision_head_=use_decision_head,
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
        :param use_decision_head: Boolean specifying whether the projection from visual features to linguistic features
            is to be done using a decision module/head or, else, a batched scalar product.
            WARNING: one should not use `embedding_dropout_prob != 0.0` when using a decision head.
        :param differentiable: Boolean specifying whether to use the differentiable graph (from loss to speaker via listener),
            or the non-differentiable graph, only updating the listener.
        """
        global use_decision_head
        use_decision_head = use_decision_head_

        super(DifferentiableObverterAgent, self).__init__(
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs)

        self.register_hook(sentence_length_entropy_logging_hook)
        
        self.kwargs = kwargs 
        self.force_eos = self.kwargs["force_eos"] if "force_eos" in self.kwargs else False

        # Differentiability?
        self.differentiable = differentiable
        if not(self.differentiable):
            self.input_stream_ids['speaker'].update({
                'sentences_one_hot':'modules:current_listener:sentences_one_hot.detach',
                'sentences_widx':'modules:current_listener:sentences_widx.detach',
            })

            self.input_stream_ids['listener'].update({
                'sentences_one_hot':'modules:current_speaker:sentences_one_hot.detach',
                'sentences_widx':'modules:current_speaker:sentences_widx.detach', 
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
                                               non_linearities=self.kwargs['cnn_encoder_non_linearities'],
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

            self.use_feat_converter = self.kwargs['use_feat_converter']
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

        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['temporal_encoder_nbr_hidden_units'])
        
        symbol_processing_input_dim = self.kwargs['symbol_embedding_size']
        self.symbol_processing = nn.GRU(input_size=symbol_processing_input_dim,
            hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
            num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
            batch_first=True,
            dropout=self.kwargs['dropout_prob'],
            bidirectional=False
        )

        """
        self.decision_head = nn.Sequential(
            nn.Linear(2*self.kwargs["symbol_processing_nbr_hidden_units"],512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256,1),
            #nn.Sigmoid()
        )
        """

        if use_decision_head:
            decision_head_input_size = self.kwargs["symbol_processing_nbr_hidden_units"]+self.encoder_feature_shape
            self.decision_head = nn.Sequential(
                nn.Linear(decision_head_input_size,128),
                #nn.BatchNorm1d(num_features=128),
                nn.Dropout(p=self.kwargs["dropout_prob"]),
                nn.ReLU(),
                nn.Linear(128, nbr_head_outputs),
                #nn.Sigmoid()
            )

        if self.use_sentences_one_hot_vectors:
            #self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
            if self.kwargs['embedding_dropout_prob'] > 0.0:
                self.symbol_encoder = nn.Sequential(
                    nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False),
                    nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
                )
            else:
                self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
        else:
            self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_embedding_size'], padding_idx=self.vocab_size)
        
        self.tau_fc = nn.Sequential(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
                                          nn.Softplus())

        self.not_target_logits_per_token = nn.Parameter(torch.ones((1, self.kwargs['max_sentence_length'], 1)))
        
        self.reset()

    def reset(self):
        # TODO: verify that initialization of decision head is not an issue:
        #self.decision_head.apply(layer_init)
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        self.embedding_tf_final_outputs = None
        self._reset_rnn_states()

    def _tidyup(self):
        """
        Called at the agent level at the end of the `compute` function.
        """
        self.embedding_tf_final_outputs = None
        
        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

    def _compute_tau(self, tau0, h):
        invtau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return invtau

    def embed_sentences(self, sentences):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :returns embedded_sentences: Tensor of shape `(batch_size, max_sentence_length, symbol_embedding_size)` containing the padded sequence of embedded symbols.
        """
        batch_size = sentences.shape[0]
        # (batch_size, max_sentence_length, self.vocab_size)

        if self.use_sentences_one_hot_vectors:
            sentences = sentences.view((-1, self.vocab_size)).float()
            # (batch_size*max_sentence_length, vocab_size)
            embedded_sentences = self.symbol_encoder(sentences)
            # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
        else:
            sentences = sentences.view((-1, 1)).long()
            # (batch_size*max_sentence_length, 1)
            embedded_sentences = self.symbol_encoder(sentences)
            # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
            if self.differentiable:
                embedded_sentences = (sentences-sentences.detach()).unsqueeze(-1)+embedded_sentences
        
        embedded_sentences = embedded_sentences.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
        # (batch_size, max_sentence_length, kwargs['symbol_embedding_size'])
        return embedded_sentences

    def find_sentence_lengths(self, sentences):
        """
        Adapted from:
        https://github.com/facebookresearch/EGG/blob/2e2d42e73f50af0ce70ab22e1ff77bf3a38ab6ef/egg/core/util.py#L267

        :param sentences:   Tensor of shape `(batch_size, max_sentence_length, vocab_size/1)` 
                            containing one-hot-encoded symbols.
        :returns: Tensor of shape `(batch_size,)` containing lengths of sentences until the first EoS token, included!
                    NOTE: we include the EoS token to guarantee non-negative sentence lenghts...

        """
        batch_size = sentences.shape[0]
        if self.use_sentences_one_hot_vectors:
            sentences_token_idx = (sentences*torch.arange(self.vocab_size).to(sentences.device)).sum(-1)
        else:
            sentences_token_idx = sentences.squeeze(-1)
        #(batch_size, max_sentence_length)
        eos_mask = (sentences_token_idx==self.vocab_stop_idx)
        
        padding_with_eos = (eos_mask.cumsum(-1).sum()>batch_size)
        # Include first EoS Symbol:
        if padding_with_eos:
            token_mask = ((eos_mask.cumsum(-1)>1)<=0)
            lengths = token_mask.sum(-1)
            #(batch_size, )
        else:
            token_mask = ((eos_mask.cumsum(-1)>0)<=0)
            lengths = token_mask.sum(-1)
            
        if not(padding_with_eos):
            # If excluding first EoS:
            lengths = lengths.add(1)
        sentence_lengths = lengths.clamp(max=self.max_sentence_length)
        #(batch_size, ) 

        return sentence_lengths

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

                if self.kwargs['use_feat_converter']:
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

    def _reason(self, sentences, features, kwargs=None):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        batch_size = features.size(0)
        nbr_distractors_po = features.size(1)
        features_dim =features.size(-1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        # Forward pass:
        if normalize and kwargs is None:
            embedding_tf_final_outputs = self.normalization(features.reshape(-1, features_dim))
            embedding_tf_final_outputs = embedding_tf_final_outputs.reshape((batch_size, nbr_distractors_po, -1))
        else:
            embedding_tf_final_outputs = features.reshape((batch_size, nbr_distractors_po, -1))

        if kwargs is None:
            self.embedding_tf_final_outputs = embedding_tf_final_outputs

        # Consume the sentences:
        max_sentence_length = sentences.shape[1]
        sentences = sentences.reshape((batch_size, max_sentence_length, -1))
        # (batch_size, max_sentence_length, self.vocab_size)
        embedded_sentences = self.embed_sentences(sentences)
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        

        sentence_lengths = self.find_sentence_lengths(sentences)
        #(batch_size, )
        if packpadding:
            packed_embedded_sentences = nn.utils.rnn.pack_padded_sequence(
                embedded_sentences, 
                sentence_lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
        else:
            packed_embedded_sentences = embedded_sentences
        
        # We initialize the rnn_states to either None, if it is not multi-round, or:
        states = None
        if kwargs is None:
            states = self.rnn_states
        elif "rnn_states" in kwargs:
            states = kwargs["rnn_states"]

        rnn_outputs, next_rnn_states = self.symbol_processing(packed_embedded_sentences, states)    
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        if kwargs is  None:
            self.rnn_states = next_rnn_states
        else:
            kwargs["next_rnn_states"] = next_rnn_states
        
        if packpadding:
            rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs,batch_first=True)
        
        # rnn_outputs is padded, so we need to propagate the real values:
        max_sl = max(sentence_lengths)
        if packpadding and any(sentence_lengths!=max_sl):
            for bidx in range(batch_size):
                sl = sentence_lengths[bidx]
                if sl == max_sl: continue
                rnn_outputs[bidx,sl-1:] = rnn_outputs[bidx,sl-1]
        
        if kwargs is not None:
            kwargs["rnn_outputs"] = rnn_outputs
        
        # Compute the decision: following each hidden/output vector from the rnn:
        decision_logits = []
        bemb = embedding_tf_final_outputs.view((batch_size, nbr_distractors_po, -1))
        # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent), cnn_encoder_feature_shape)
        
        bemb = embedding_tf_final_outputs.view((batch_size*nbr_distractors_po, -1))
        # (batch_size*nbr_distractors_po, cnn_encoder_feature_shape)
        
        
        for widx in range(rnn_outputs.size(1)):
            decision_inputs = rnn_outputs[:,widx,...].unsqueeze(1).repeat(1, nbr_distractors_po, 1)
            # (batch_size, nbr_distractors_po, kwargs['symbol_processing_nbr_hidden_units'])
            decision_inputs = decision_inputs.reshape(batch_size*nbr_distractors_po, -1)
            # (batch_size*nbr_distractors_po, kwargs['symbol_processing_nbr_hidden_units'])
            
            decision_head_input = torch.cat([decision_inputs, bemb], dim=-1)
            # (batch_size*nbr_distractors_po, 2*kwargs['symbol_processing_nbr_hidden_units'])
            
            decision_logits_until_widx = self.decision_head(decision_head_input).reshape((batch_size, nbr_distractors_po, nbr_head_outputs))
            # Linear output...
            # (batch_size, nbr_distractors_po, nbr_head_outputs)
                
            decision_logits.append(decision_logits_until_widx.unsqueeze(1))
            # (batch_size, 1, (nbr_distractors+1) 
            # / ? (descriptive mode depends on the role of the agent),
            # nodim / nbr_head_outputs )
        decision_logits = torch.cat(decision_logits, dim=1)
        # (batch_size, max_sentence_length, (nbr_distractors+1)
        # / ? (descriptive mode depends on the role of the agent),
        # nodim / nbr_head_outputs )
            
        if self.kwargs['descriptive']: # or kwargs is not None
            possible_targets = decision_logits[...,0]
            # (batch_size, max_sentence_length, (nbr_distractors+1), )
            not_target = decision_logits[...,1].max(dim=-1, keepdim=True)[0]
            # (batch_size, max_sentence_length, 1)                
            decision_logits = torch.cat([possible_targets, not_target], dim=-1 )
            # (batch_size, max_sentence_length, (nbr_distractors+2))
        
        # NOW: Regularization to make those values actual log probabilities...
        decision_logits = torch.log_softmax(decision_logits, dim=-1)
        
        return decision_logits, embedding_tf_final_outputs
        #NOMORE: If use_decision_head, decision_logits is actually the probabilities...
        # NOW: actual logits:log_softmax...


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
        features_dim = features.size(-1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, cnn_encoder_feature_dim)
        
        # Forward pass:
        if normalize:
            self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, features_dim)))
            self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, nbr_distractors_po, -1))
            # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])
        else:
            self.embedding_tf_final_outputs = features.reshape(batch_size, nbr_distractors_po, -1)

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
        """
        wl_next_sentences_hidden_states, \
        wl_next_sentences_widx, \
        wl_next_sentences_logits, \
        wl_next_sentences_one_hots = DifferentiableObverterAgent._compute_sentence(
            features_embedding=self.embedding_tf_final_outputs,
            target_idx=predicted_target_idx,
            symbol_encoder=self.symbol_encoder,
            symbol_processing=self.symbol_processing,
            init_rnn_states=None, #self.rnn_states,
            allowed_vocab_size=allowed_vocab_size,
            vocab_size=self.vocab_size,
            max_sentence_length=self.max_sentence_length,
            nbr_distractors_po=nbr_distractors_po,
            rnn_output_size=self.kwargs["symbol_processing_nbr_hidden_units"],
            operation=operation,
            force_eos=self.force_eos,
            vocab_stop_idx=self.vocab_stop_idx,
            use_obverter_threshold_to_stop_message_generation=self.kwargs['use_obverter_threshold_to_stop_message_generation'],
            use_stop_word=use_stop_word_in_compute_sentence,
            _compute_tau=self._compute_tau,
            not_target_logits_per_token=self.not_target_logits_per_token if self.use_learning_not_target_logit else None,
            use_sentences_one_hot_vectors=self.use_sentences_one_hot_vectors,
            training=self.training,
            agent=self,
            logger=self.logger,
            whole_sentence=True)
        """


        next_sentences_hidden_states, \
        next_sentences_widx, \
        next_sentences_logits, \
        next_sentences_one_hots = DifferentiableObverterAgent._compute_sentence(
            features_embedding=self.embedding_tf_final_outputs,
            target_idx=predicted_target_idx,
            symbol_encoder=self.symbol_encoder,
            symbol_processing=self.symbol_processing,
            init_rnn_states=None, #self.rnn_states,
            allowed_vocab_size=allowed_vocab_size,
            vocab_size=self.vocab_size,
            max_sentence_length=self.max_sentence_length,
            nbr_distractors_po=nbr_distractors_po,
            rnn_output_size=self.kwargs["symbol_processing_nbr_hidden_units"],
            operation=operation,
            force_eos=self.force_eos,
            vocab_stop_idx=self.vocab_stop_idx,
            use_obverter_threshold_to_stop_message_generation=self.kwargs['use_obverter_threshold_to_stop_message_generation'],
            use_stop_word=use_stop_word_in_compute_sentence,
            _compute_tau=self._compute_tau,
            not_target_logits_per_token=self.not_target_logits_per_token if self.use_learning_not_target_logit else None,
            use_sentences_one_hot_vectors=self.use_sentences_one_hot_vectors,
            training=self.training,
            agent=self,
            logger=self.logger)

        return next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences_one_hots, self.embedding_tf_final_outputs

    def _compute_sentence(features_embedding, 
                          target_idx, 
                          symbol_encoder, 
                          symbol_processing, 
                          init_rnn_states=None,
                          allowed_vocab_size=10, 
                          vocab_size=10, 
                          max_sentence_length=14,
                          nbr_distractors_po=1,
                          rnn_output_size=256,
                          operation=torch.max,
                          force_eos=False,
                          vocab_stop_idx=0,
                          use_obverter_threshold_to_stop_message_generation=False,
                          use_stop_word=False,
                          _compute_tau=None,
                          not_target_logits_per_token=None,
                          use_sentences_one_hot_vectors=False,
                          training=False,
                          agent=None,
                          logger=None):
                          #whole_sentence=False):
        """
        Compute sentences using the obverter approach, adapted to referential game variants following the
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
        :param init_rnn_states: None or Tuple of Tensors to initialize the symbol_processing's rnn states.
        :param vocab_size: int, size of the vocabulary.
        :param max_sentence_length: int, maximal length for each generated sentences.
        :param nbr_distractors_po: int, number of distractors and target, i.e. `nbr_distractors+1.
        :param rnn_output_size: int, nbr cells in the symbol processing unit.
        :param operation: Function, expect `torch.max` or `torch.min`.
        :param force_eos: boolean specifying whether to force eos symbol at end of sentences.
        :param vocab_stop_idx: int, index of the STOP symbol in the vocabulary.
        :param use_obverter_threshold_to_stop_message_generation:  boolean, or float that specifies whether to stop the 
                                                                    message generation when the decision module's 
                                                                    output probability is abobe a given threshold 
                                                                    (or below it if the operation is `torch.min`).
                                                                    If it is a float, then it is the value of the threshold.
        :param use_stop_word: boolean that specifies whether to use one of the word in the vocabulary with a pre-defined meaning,
                              that is that it is a STOP token, thus effictively ending the symbol generation for the current sentence.
        :param training: boolean specifying whether to use training sampling method or testing (argmax...).
        
        :returns:
            - sentences_widx: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], 1)` where `b` is the batch index.
                             It represents the indices of the chosen words.
            - sentences_logits: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the logits of words over the decision module's potential to choose the target experience as output.
            - sentences_one_hots: List[Tensor] of length `batch_size` with shapes `(1, sentences_lenght[b], vocab_size)` where `b` is the batch index.
                                It represents the sentences as one-hot-encoded word vectors.
        
        """
        batch_size = features_embedding.size(0)
        symbol_processing_nbr_hidden_units = rnn_output_size 
        cnn_encoder_feature_shape = features_embedding.size(-1)
        
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
        
        vocab_idx = vocab_idx.to(features_embedding.device)

        embedded_vocab = symbol_encoder(vocab_idx).view((allowed_vocab_size,1,-1))
        # Embedding: (batch_size=allowed_vocab_size, 1, kwargs['symbol_embedding_size'])
        embedded_vocab = embedded_vocab.unsqueeze(0).repeat((batch_size,1,1,1))
        # (batch_size, allowed_vocab_size, 1, kwargs['symbol_embedding_size'])
        embedded_vocab = embedded_vocab.reshape(batch_size*allowed_vocab_size, 1, -1)
        # (batch_size*allowed_vocab_size, 1, kwargs['symbol_embedding_size'])
        
        vocab_idx = vocab_idx.reshape((allowed_vocab_size,1,-1))
        # (allowed_vocab_size, 1, vocab_size)
        vocab_idx = vocab_idx.unsqueeze(0).repeat((batch_size,1,1,1))
        # (batch_size, allowed_vocab_size, 1, vocab_size)
        vocab_idx = vocab_idx.reshape((batch_size*allowed_vocab_size,-1))
        # (batch_size*allowed_vocab_size, vocab_size)
        
        sentences_hidden_states = list()
        sentences_widx = vocab_stop_idx*torch.ones((batch_size, max_sentence_length, 1)).to(features_embedding.device).long() #list()
        sentences_logits = torch.zeros((batch_size, max_sentence_length, vocab_size)).to(features_embedding.device) #list()
        # PREVIOUSLY:
        #sentences_one_hots = torch.zeros((batch_size, max_sentence_length, vocab_size)).to(features_embedding.device) #list()
        # NOW: filling up the sentence with eos token:
        sentences_one_hots = torch.zeros((batch_size, max_sentence_length, vocab_size)).to(features_embedding.device) #list()
        # (batch_size, max_sentence_length, vocab_size)
        sentences_one_hots[..., vocab_stop_idx] = 1.0

        bemb = features_embedding.reshape((batch_size, nbr_distractors_po, -1))
        # (batch_size, (nbr_distractors+1), cnn_encoder_feature_shape)

        if inner_model: #bmm:
            bemb = bemb.unsqueeze(1).repeat((1, allowed_vocab_size, 1, 1))
            # (batch_size, allowed_vocab_size, (nbr_distractors+1), kwargs['temporal_encoder_nbr_hidden_units'])
            bemb = bemb.reshape(batch_size*allowed_vocab_size, nbr_distractors_po, -1)
        
        btarget_idx = target_idx.reshape(batch_size, 1, 1)
        # target_idx: (batch_size, 1, 1)
        btarget_idx = btarget_idx.repeat(1, allowed_vocab_size, 1)
        # (batch_size, allowed_vocab_size, 1)
        
        msl = max_sentence_length
        if force_eos:
            msl -= 1
        
        initial_token_logits = torch.zeros(batch_size, vocab_size)
        initial_full_vocab_rnn_outputs = torch.zeros(batch_size, vocab_size, symbol_processing_nbr_hidden_units)
        # (batch_size, vocab_size, hidden_size)

        # Generation can be stopped if the confidence is above a given threshold.
        # In that case, we record eos symbols at every following token position.
        mask_record_generation = torch.ones(batch_size,1).to(bemb.device)

        for token_idx in range(msl):
            if all(mask_record_generation == torch.zeros_like(mask_record_generation)):
                break
            if states is not None:
                '''
                hs, cs = states[0], states[1]
                hs = hs.repeat( 1, vocab_size, 1)
                cs = cs.repeat( 1, vocab_size, 1)
                rnn_states = (hs, cs)
                '''
                rnn_states = states.repeat(1,allowed_vocab_size, 1)
                # (1, batch_size*allowed_vocab_size, hidden_size)
            else :
                rnn_states = states

            
            if inner_model:
                if whole_sentence:
                    if use_sentences_one_hot_vectors:
                        sentences = sentences_one_hots.unsqueeze(1).repeat(1, allowed_vocab_size, 1, 1)
                        # (batch_size, allowed_vocab_size, max_sentence_length, vocab_size)
                        sentences = sentences.reshape(batch_size*allowed_vocab_size, max_sentence_length, -1)
                        # (batch_size*allowed_vocab_size, max_sentence_length, vocab_size)
                    else:
                        sentences = sentences_widx.unsqueeze(1).repeat(1, allowed_vocab_size, 1, 1)
                        # (batch_size, allowed_vocab_size, max_sentence_length, 1)
                        sentences = sentences.reshape(batch_size*allowed_vocab_size, max_sentence_length, -1)
                        # (batch_size*allowed_vocab_size, max_sentence_length, 1)
                    sentences[:,token_idx] = vocab_idx
                    # (batch_size*allowed_vocab_size, max_sentence_length, 1)
                    # or (batch_size*allowed_vocab_size, max_sentence_length, vocab_size) if use_sentences_one_hot_vectors
                    # vocab_idx is of the required shape.
                    kwargs = {"rnn_states":None}
                else:
                    sentences = vocab_idx.unsqueeze(1)
                    # (batch_size*allowed_vocab_size, sentence_length=1, 1 / 10 if one_hot)
                    kwargs = {"rnn_states":rnn_states}

                decision_logits, _ = agent._reason(sentences=sentences,features=bemb, kwargs=kwargs)
                # (batch_size*allowed_vocab_size, max_sentence_length/1, nbr_distractors_po)
                #NOMORE: Actually a probability:
                #NOMORE: Output of Sigmoid if use_decision_head and nbr head outputs ==1, or softmax if nbr_head_outputs ===2...
                # NOW: actual logits: log_softmax output
                next_rnn_states = kwargs["next_rnn_states"]
                # (1, batch_size*allowed_vocab_size, hidden_size)
                rnn_outputs = kwargs["rnn_outputs"]

                rnn_outputs = rnn_outputs.reshape(batch_size, allowed_vocab_size, -1, symbol_processing_nbr_hidden_units)
                # (batch_size, allowed_vocab_size, sentence_length, symbol_processing_nbr_hidden_units)
                
                if whole_sentence:
                    rnn_outputs = rnn_outputs[:,:,token_idx,...]
                    # (batch_size, allowed_vocab_size, symbol_processing_nbr_hidden_units)
                    decision_logits = decision_logits[:,token_idx]
                    # (batch_size*allowed_vocab_size, nbr_distractors_po)
                    #NOW: next_rnn_states = rnn_outputs.unsqueeze(0)
                    # (1, batch_size*allowed_vocab_size, hidden_size)                    
                else:
                    # Selecting only the last element of the sequence, but technically it should be the only one:
                    rnn_outputs = rnn_outputs[:,:,-1,...]
                    # (batch_size, allowed_vocab_size, symbol_processing_nbr_hidden_units)
                    decision_logits = decision_logits[:,-1]
                    # (batch_size*allowed_vocab_size, nbr_distractors_po)

                decision_logits = decision_logits.reshape(batch_size, allowed_vocab_size, -1)
                # NOW actual log softmax logit
                # (batch_size, allowed_vocab_size, nbr_distractors_po)
            
            if use_decision_head or use_one_minus_max_prob:
                #inner_model_output_probs = decision_logits.reshape(batch_size, allowed_vocab_size, -1)
                
                # PREVIOUSLY:
                #inner_model_output_probs = decision_logits.reshape(batch_size, allowed_vocab_size, -1).softmax(dim=-1)
                # NOW:
                inner_model_output_probs = torch.exp(decision_logits.reshape(batch_size, allowed_vocab_size, -1))
                
                target_decision_logits = decision_logits.gather(index=btarget_idx, dim=-1).reshape((batch_size, allowed_vocab_size))
                # (batch_size, allowed_vocab_size, )
                target_decision_probs = inner_model_output_probs.gather(index=btarget_idx, dim=-1).reshape((batch_size, allowed_vocab_size))
                # (batch_size, allowed_vocab_size, )
            
            # values for each token are now compared: 
            # the token that yield the greatest advantage over not-stimulus will be chosen
            if LogSoftmaxAfterLogOnSigmoid :
                target_decision_logits = target_decision_logits.log_softmax(dim=-1)
            # (batch_size, allowed_vocab_size, )
            
            token_logits = reg_nan(target_decision_logits)
            token_probs = reg_nan(target_decision_probs)
            if use_one_minus_max_prob:
                # Log of probabilities of the token yielding the target stimulus.
                # but they are not summing to one over all the tokens...
                # They are individual bernoulli  probabilities,
                # so we treat them like logits when considering them toker for sampling:
                token_dist = torch.distributions.Categorical(logits=token_logits)
            else:
                token_dist = torch.distributions.Categorical(logits=token_logits)
                #token_dist_with_probs = torch.distributions.Categorical(probs=token_probs)
            # (batch_size, vocab_size)
            
            if (not_always_argmax and agent.training) or always_categorical_sampling:
                sampled_token = token_dist.sample().reshape(-1,1)
                # (batch_size,1)
            else:
                # Previously:
                #sampled_token = token_logits.argmax(dim=-1).reshape(-1,1)
                # NOW: with probs, like in benbogin implem
                sampled_token = token_probs.argmax(dim=-1).reshape(-1,1)
                sampled_token_prob = token_probs.max(dim=-1)[0].reshape(-1,1)
                # (batch_size,1)

            
            if allowed_vocab_size < vocab_size:
                raise NotImplementedError
                token_logits = initial_token_logits.to(target_decision_logits.device)
                # (batch_size, vocab_size, )
                token_logits[:, :allowed_vocab_size] = target_decision_logits
                #token_logits = torch.log_softmax(token_logits, dim=-1)
                # (batch_size, vocab_size, )

            
            if use_decision_head or use_one_minus_max_prob: 
                # (batch_size, allowed_vocab_size, nbr_distractors_po )
                # Previously:
                """
                token_probs = inner_model_output_probs.gather(index=btarget_idx, dim=-1).reshape((batch_size, allowed_vocab_size))
                if torch.any(btarget_idx!=0):
                   import ipdb; ipdb.set_trace()
                
                # (batch_size, allowed_vocab_size, )
                sampled_token_prob = token_probs.gather(index=sampled_token.long(), dim=-1)
                # (batch_size,1)
                """
                # NOW:
                token_probs = sampled_token_prob
                # (batch_size,1)
                
                token_logits = reg_nan(token_probs.log())
                # (batch_size,1)
                
            # Filter for batch element that are still being generated:
            # If we are no longer generating token for a given batch element, then we sample SoS token:
            masked_sampled_token = sampled_token*mask_record_generation + (1-mask_record_generation)*vocab_stop_idx
            # If we are no longer generating token for a given batch element, then the logit are zero everywhere:
            masked_token_logits = token_logits*mask_record_generation + scale_negative_logit*(1-mask_record_generation)
            
            sentences_widx[:,token_idx] = masked_sampled_token
            # (batch_size, 1)
            sentences_logits[:,token_idx] = masked_token_logits
            # (batch_size, vocab_size)

            assert all(masked_sampled_token < vocab_size)
            token_one_hot = nn.functional.one_hot(
                masked_sampled_token.long(), 
                num_classes=vocab_size).view((-1, vocab_size))
            # (batch_size, vocab_size)
            #sentences_one_hots.append(token_one_hot)
            # (batch_size, vocab_size)
            sentences_one_hots[:,token_idx] = token_one_hot
            # (batch_size, max_sentence_length, vocab_size)
            
            full_vocab_rnn_outputs = initial_full_vocab_rnn_outputs.to(token_logits.device)
            # (batch_size, vocab_size, hidden_size)
            full_vocab_rnn_outputs[:,:allowed_vocab_size,...] = rnn_outputs.reshape((batch_size, allowed_vocab_size, -1))
            # (batch_size, vocab_size, hidden_size)
            sentences_hidden_states.append(full_vocab_rnn_outputs)
            # (batch_size, vocab_size, hidden_size)
        

            #Bookkeeping:
            ## next rnn_states:
            #states = [st[-1, sampled_token].view((1,1,-1)) for st in next_rnn_states]

            # Selecting over hidden_layer*directions: then reshaping:
            #PREVIOUSLY, to handle whole setence: states = next_rnn_states[-1].reshape((batch_size, allowed_vocab_size, -1))
            #NOW:  (1, batch_size*allowed_vocab_size, hidden_size)
            states = next_rnn_states.reshape((batch_size, allowed_vocab_size, -1))
            # (batch_size, allowed_vocab_size, kwargs['symbol_processing_nbr_hidden_units'])
            states = states.gather(
                index=sampled_token.unsqueeze(-1).repeat(1, 1, symbol_processing_nbr_hidden_units).long(), 
                dim=1
            ).view((1, batch_size, -1))
            # (1, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
            
            ## Mask controlling whether we record the following token generated or not:
            if use_obverter_threshold_to_stop_message_generation:
                if operation == torch.max:
                    operation_condition = (sampled_token_prob >= use_obverter_threshold_to_stop_message_generation)
                else:
                    raise NotImplementedError
                    #operation_condition = (sampled_token_prob < 1-use_obverter_threshold_to_stop_message_generation) 
                
                mask_record_generation *= (1-operation_condition.float())
                
            if use_stop_word:
                stop_word_sampled = (sampled_token.long() == vocab_stop_idx)
                mask_record_generation *= (1-stop_word_sampled.float())
                
        if force_eos:
            eos_idx = vocab_stop_idx*torch.ones_like(sentences_widx[0])
            # (batch_size, 1)
            sentences_widx.append(eos_idx)
            
            eos_one_hot = nn.functional.one_hot(
                eos_idx.view(-1,1).long(), 
                num_classes=vocab_size).view((-1, vocab_size))
            # (batch_size, vocab_size)
            #sentences_one_hots.append(eos_one_hot)
            sentences_one_hots[:,token_idx+1:,...] = eos_one_hot.unsqueeze(1)
            
            zeros = torch.zeros_like(sentences_logits[0])
            # (batch_size, vocab_size)
            sentences_logits.append(zeros)

            sentences_hidden_states.append(torch.zeros_like(sentences_hidden_states[-1]))

        sentences_hidden_states = torch.stack(sentences_hidden_states, dim=1)
        # (batch_size, max_sentence_length, vocab_size, kwargs['symbol_preprocessing_nbr_hidden_units'])
        
        #sentences_one_hots = torch.stack(sentences_one_hots, dim=1)
        # (batch_size, max_sentence_length, vocab_size)

        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots


    