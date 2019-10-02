import torch
import torch.nn as nn
import torch.nn.functional as F

from .listener import Listener
from ..networks import choose_architecture, layer_init
from ..utils import gumbel_softmax, permutate_latents


class DifferentiableObverterAgent(Listener):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='o0', logger=None, use_sentences_one_hot_vectors=False):
        """
        :param obs_shape: tuple defining the shape of the experience following `(nbr_distractors+1, nbr_stimulus, *experience_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        """
        super(DifferentiableObverterAgent, self).__init__(obs_shape, vocab_size, max_sentence_length, agent_id, logger, kwargs)
        self.kwargs = kwargs 
        self.use_sentences_one_hot_vectors = use_sentences_one_hot_vectors
        self.use_learning_not_target_logit = True

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
        elif 'ResNet18' in self.kwargs['architecture'] and not("MHDPA" in self.kwargs['architecture']):
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'])
        elif 'ResNet18' in self.kwargs['architecture'] and "MHDPA" in self.kwargs['architecture']:
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                  dropout=self.kwargs['dropout_prob'],
                                                  MHDPANbrHead=self.kwargs['mhdpa_nbr_head'],
                                                  MHDPANbrRecUpdate=self.kwargs['mhdpa_nbr_rec_update'],
                                                  MHDPANbrMLPUnit=self.kwargs['mhdpa_nbr_mlp_unit'],
                                                  MHDPAInteractionDim=self.kwargs['mhdpa_interaction_dim'])
        elif self.kwargs['architecture'] == 'CNN-MHDPA':
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  hidden_units_list=None,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                  nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                                  kernels=self.kwargs['cnn_encoder_kernels'],
                                                  strides=self.kwargs['cnn_encoder_strides'],
                                                  paddings=self.kwargs['cnn_encoder_paddings'],
                                                  dropout=self.kwargs['dropout_prob'],
                                                  MHDPANbrHead=self.kwargs['mhdpa_nbr_head'],
                                                  MHDPANbrRecUpdate=self.kwargs['mhdpa_nbr_rec_update'],
                                                  MHDPANbrMLPUnit=self.kwargs['mhdpa_nbr_mlp_unit'],
                                                  MHDPAInteractionDim=self.kwargs['mhdpa_interaction_dim'])
        elif 'BetaVAE' in self.kwargs['architecture']:
            self.VAE_losses = list()
            self.neg_log_likelyhoods = list()
            self.kl_divs = list()
            self.kl_div_regs = list()
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                   kwargs=self.kwargs,
                                                   input_shape=cnn_input_shape,
                                                   feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                   dropout=self.kwargs['dropout_prob'])
            self.VAE = self.cnn_encoder

            self.TC_losses = list()
            self.vae_modularity = list()
            if 'vae_tc_discriminator_hidden_units' in self.kwargs:
                self.tc_discriminator = choose_architecture(architecture='MLP',
                                                            input_shape=[self.VAE.latent_dim],
                                                            hidden_units_list=self.kwargs['vae_tc_discriminator_hidden_units'])
            else:
                self.tc_discriminator = None



        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        self.temporal_feature_encoder = nn.GRU(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=self.kwargs['dropout_prob'],
                                          bidirectional=False)

        symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']
        self.symbol_processing = nn.GRU(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)

        if self.use_sentences_one_hot_vectors:
            self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False)
        else:
            self.symbol_encoder = nn.Embedding(self.vocab_size+2, self.kwargs['symbol_processing_nbr_hidden_units'], padding_idx=self.vocab_size)
        
        self.symbol_decoder = nn.ModuleList()
        self.symbol_decoder.append(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size))
        if self.kwargs['dropout_prob']: self.symbol_decoder.append(nn.Dropout(p=self.kwargs['dropout_prob']))
        
        self.tau_fc = layer_init(nn.Linear(self.kwargs['temporal_encoder_nbr_hidden_units'], 1 , bias=False))
        
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1,self.kwargs['max_sentence_length'])))
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
        if 'BetaVAE' in self.kwargs['architecture']:
            self.VAE_losses = list()
            self.neg_log_likelyhoods.clear()
            self.kl_divs.clear()
            self.kl_div_regs.clear()
            if self.tc_discriminator is not None:
                self.TC_losses = list()
                self.vae_modularity.clear()

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
            if 'BetaVAE' in self.kwargs['architecture']:
                VAE_loss, neg_log_lik, kl_div_reg, kl_div  = self.cnn_encoder.compute_loss(stin)
                self.VAE_losses.append(VAE_loss)
                self.neg_log_likelyhoods.append(neg_log_lik.cpu())
                self.kl_divs.append(kl_div.cpu())
                self.kl_div_regs.append(kl_div_reg.cpu())
                featout = self.cnn_encoder.mu 

                if self.tc_discriminator is not None:
                    z = self.cnn_encoder.z.detach()
                    z1, z2 = z.chunk(2, dim=0)
                    
                    zeros = torch.zeros(z1.size(0)).long().to(z1.device)
                    ones = torch.ones(z2.size(0)).long().to(z2.device)
                    
                    pz = permutate_latents(z2)
                    Dz = self.tc_discriminator(z1)
                    Dpz = self.tc_discriminator(pz)
                    tc_l11 = 0.5*F.cross_entropy(input=Dz, target=zeros, reduction='none')
                    # (b1, )
                    tc_l12 = 0.5*F.cross_entropy(input=Dpz, target=ones, reduction='none')
                    # (b2, )
                    
                    zeros = torch.zeros(z2.size(0)).long().to(z2.device)
                    ones = torch.ones(z1.size(0)).long().to(z1.device)
                    
                    pz = permutate_latents(z1)
                    Dz = self.tc_discriminator(z2)
                    Dpz = self.tc_discriminator(pz)
                    tc_l21 = 0.5*F.cross_entropy(input=Dz, target=zeros, reduction='none')
                    # (b1, )
                    tc_l22 = 0.5*F.cross_entropy(input=Dpz, target=ones, reduction='none')
                    # (b2, )
                    
                    tc_loss1 = tc_l11 + tc_l22
                    tc_loss2 = tc_l12 + tc_l21
                    
                    tc_loss = torch.cat([tc_loss1, tc_loss2])
                    # (b, )
                    self.TC_losses.append(tc_loss)
                    
                    probDz = F.softmax(Dz.detach(), dim=1)[...,:1]
                    probDpz = F.softmax(Dpz.detach(), dim=1)[...,1:]
                    discr_acc = (torch.cat([probDz,probDpz],dim=0) >= 0.5).sum().float().div(2*probDz.size(0))
                    self.vae_modularity.append(discr_acc.cpu().unsqueeze(0))
            else:
                featout = self.cnn_encoder(stin)
            features.append( featout)
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['cnn_encoder_feature_dim'])
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        if 'BetaVAE' in self.kwargs['architecture']:
            self.VAE_losses = torch.cat(self.VAE_losses).contiguous().view((batch_size,-1)).mean(dim=-1)
            self.log_dict['kl_div'] = torch.cat(self.kl_divs).mean()
            self.log_dict['kl_div_reg'] = torch.cat(self.kl_div_regs).mean()
            self.log_dict['kl_capacity'] = torch.Tensor([100.0*self.cnn_encoder.EncodingCapacity/self.cnn_encoder.maxEncodingCapacity])
            self.log_dict['neg_log_lik'] = torch.cat(self.neg_log_likelyhoods).mean()
            if self.tc_discriminator is not None:
                self.TC_losses = torch.cat(self.TC_losses).contiguous().view((batch_size,-1)).mean(dim=-1)
                self.log_dict['Modularity'] = torch.cat(self.vae_modularity).mean()

        return features 

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
        features = features.view(-1, *(features.size()[2:]))
        # (batch_size*(nbr_distractors+1) / ? (descriptive mode depends on the role of the agent), nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
        rnn_outputs = []
        total_size = features.size(0)
        mini_batch_size = min(self.kwargs['temporal_encoder_mini_batch_size'], total_size)
        for featin in torch.split(features, split_size_or_sections=mini_batch_size, dim=0):
            outputs, _ = self.temporal_feature_encoder(featin)
            # (mini_batch_size, -1, kwargs['temporal_encoder_nbr_hidden_units'])
            rnn_outputs.append( outputs)
        outputs = torch.cat(rnn_outputs, dim=0)
        outputs = outputs.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['temporal_encoder_nbr_hidden_units'])
        
        embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
        self.embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size, -1)
        # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) * kwargs['temporal_encoder_nbr_hidden_units'])
        
        # Consume the sentences:
        if self.use_sentences_one_hot_vectors:
            sentences = sentences.view((-1, self.vocab_size))
            # (batch_size*max_sentence_lengthm, vocab_size)
            diff_embedded_sentences = self.symbol_encoder(sentences)
            diff_embedded_sentences = diff_embedded_sentences.view((batch_size, -1, self.kwargs['symbol_processing_nbr_hidden_units']))
            # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        else:
            sentences = sentences.view((batch_size, -1))
            # (batch_size, max_sentence_length)
            embedded_sentences = self.symbol_encoder(sentences.long())
            #TODO: decide on a differentiation strategy: sum or product?
            #diff_embedded_sentences = (sentences/sentences.detach()).unsqueeze(-1)*embedded_sentences
            diff_embedded_sentences = (sentences-sentences.detach()).unsqueeze(-1)+embedded_sentences    
            diff_embedded_sentences = diff_embedded_sentences.view((batch_size, -1, self.kwargs['symbol_processing_nbr_hidden_units']))
            # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        
        states = self.rnn_states
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        rnn_outputs, self.rnn_states = self.symbol_processing(diff_embedded_sentences, states)          
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
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
        
        
        if not(self.use_learning_not_target_logit):
            l_shape = decision_logits.size()
            not_target_logit = torch.zeros( *l_shape[:2], 1)
        else:
            not_target_logit = self.not_target_logits_per_token.view((1,self.max_sentence_length,1)).repeat(batch_size, 1, 1)
        if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
        decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
        
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
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        if self.embedding_tf_final_outputs is None: 
            features = features.view(-1, *(features.size()[2:]))
            # (batch_size*(nbr_distractors+1) / ? (descriptive mode depends on the role of the agent), nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
            rnn_outputs = []
            total_size = features.size(0)
            mini_batch_size = min(self.kwargs['temporal_encoder_mini_batch_size'], total_size)
            for featin in torch.split(features, split_size_or_sections=mini_batch_size, dim=0):
                outputs, _ = self.temporal_feature_encoder(featin)
                # (mini_batch_size, -1, kwargs['temporal_encoder_nbr_hidden_units'])
                rnn_outputs.append( outputs)
            outputs = torch.cat(rnn_outputs, dim=0)
            outputs = outputs.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['temporal_encoder_nbr_hidden_units'])
        
            embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
            self.embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size, -1)
            # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) * kwargs['temporal_encoder_nbr_hidden_units'])
        
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
        next_sentences_widx, \
        next_sentences_logits, \
        next_sentences_one_hots = DifferentiableObverterAgent._compute_sentence(features_embedding=self.embedding_tf_final_outputs,
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

        return next_sentences_widx, next_sentences_logits, next_sentences_one_hots, self.embedding_tf_final_outputs

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
        '''
        vocab_idx = torch.zeros((vocab_size,vocab_size))
        for i in range(vocab_size): vocab_idx[i,i] = 1
        if features_embedding.is_cuda: vocab_idx = vocab_idx.cuda()
        vocab_idx = symbol_encoder(vocab_idx).unsqueeze(1)
        # (batch_size=vocab_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
        '''
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
        # Embedding: (batch_size=allowed_vocab_size, 1, kwargs['symbol_processing_nbr_hidden_units'])

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
                
                '''
                if nbr_distractors_po==1:
                    # Partial observability:
                    decision_probs = torch.sigmoid(decision_logits)
                else:
                    # Full observability:
                    decision_probs = F.softmax(decision_logits, dim=-1)
                # (batch_size=allowed_vocab_size, (nbr_distractors+1) )
                
                '''
                if not_target_logits_per_token is None:
                    not_target_logit = torch.zeros(decision_logits.size(0), 1)
                else:
                    not_target_logit = not_target_logits_per_token[:,sentence_token_count-1].repeat(allowed_vocab_size,1)
                if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
                decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
                
                tau0 = 1e1
                # Probs over Distractors and Vocab: 
                decision_probs = F.softmax( decision_logits.view(-1), dim=-1).view((allowed_vocab_size, -1))
                decision_probs_least_effort = F.softmax( decision_logits.view(-1)*tau0, dim=-1).view((allowed_vocab_size, -1))
                '''
                decision_probs = F.softmax( decision_logits, dim=-1)
                decision_probs_least_effort = F.softmax( decision_logits*tau0, dim=-1)
                '''
                # (batch_size=vocab_size, (nbr_distractors+2) )
                
                '''
                target_decision_probs_per_vocab_logits = decision_probs[:,btarget_idx]
                # (batch_size=vocab_size, )
                vocab_idx_op, vocab_idx_argop = operation(target_decision_probs_per_vocab_logits, dim=0)
                # (batch_size=vocab_size, )
                
                '''
                target_decision_probs_per_vocab_logits = decision_probs[:,btarget_idx]
                target_decision_probs_least_effort_per_vocab_logits = decision_probs_least_effort[:,btarget_idx]
                # (batch_size=allowed_vocab_size, )
                tau = 1.0/5e0 
                if _compute_tau is not None:    tau = _compute_tau(tau0=tau, emb=bemb[btarget_idx].unsqueeze(0))
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
                    
                    target_decision_probs_least_effort_per_vocab_logits = torch.cat([target_decision_probs_least_effort_per_vocab_logits,
                                                                                     zeros4complete_vocab], dim=0)
                vocab_idx_argop = torch.sum(arange_vocab*one_hot_sampled_vocab)
                vocab_idx_op = target_decision_probs_least_effort_per_vocab_logits[vocab_idx_argop.long()]
                
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
                if False:#use_sentences_one_hot_vectors:
                    sentences_widx[b].append((vocab_stop_idx)*torch.ones_like(vocab_idx_argop))
                else:
                    sentences_widx[b].append((vocab_size)*torch.ones_like(vocab_idx_argop))

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

        return sentences_widx, sentences_logits, sentences_one_hots
