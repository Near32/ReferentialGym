import torch
import torch.nn as nn
import torch.nn.functional as F

from .listener import Listener
from ..networks import choose_architecture, layer_init, MHDPA_RN
from ..utils import gumbel_softmax


class DifferentiableRelationalObverterAgent(Listener):
    def __init__(self,
                 kwargs, 
                 obs_shape, 
                 vocab_size=100, 
                 max_sentence_length=10, 
                 agent_id='o0', 
                 logger=None):
        """
        :param obs_shape: tuple defining the shape of the experience following `(nbr_distractors+1, nbr_stimulus, *experience_shape)`
                          where, by default, `nbr_distractors=1` and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        """
        super(DifferentiableRelationalObverterAgent, self).__init__(obs_shape, vocab_size, max_sentence_length, agent_id, logger, kwargs)
        self.kwargs = kwargs 
        self.use_sentences_one_hot_vectors = True

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
        
        self.kwargs['thought_space_depth_dim'] = self.kwargs['cnn_encoder_channels'][-1]
        self.visual_feat_spatial_dim = self.cnn_encoder.feat_map_dim
        self.mm_ponderer_depth_dim = self.kwargs['thought_space_depth_dim']+4
        # feat_depth_dim + X coord + Y coord + T coord + Vocab Coord
        self.nbr_visual_entity =  self.visual_feat_spatial_dim*self.visual_feat_spatial_dim*self.kwargs['nbr_stimulus']
        self.nbr_symbolic_entity =  self.kwargs['max_sentence_length']
        self.mm_ponderer_nbr_entity = self.nbr_visual_entity + self.nbr_symbolic_entity
        self.mm_ponderer = MHDPA_RN(output_dim=None,
                                    depth_dim=self.mm_ponderer_depth_dim,
                                    nbrHead=self.kwargs['ponderer_nbr_head'], 
                                    nbrRecurrentSharedLayers=self.kwargs['ponderer_nbr_rec_update'], 
                                    nbrEntity=self.mm_ponderer_nbr_entity,  
                                    units_per_MLP_layer=self.kwargs['ponderer_nbr_mlp_units'],
                                    interactions_dim=self.kwargs['ponderer_interactions_dim'],
                                    dropout_prob=self.kwargs['dropout_prob'])
        
        self.nbr_distractors_po = self.kwargs['nbr_distractors']+1
        self.visualXYDTS = None
        self.symbolXYDTS = None 

        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['thought_space_depth_dim'], bias=False)

        self.tau_fc = layer_init(nn.Linear(self.mm_ponderer_depth_dim, 1 , bias=False))
        
        self.not_target_logits_per_token = nn.Parameter(torch.ones((1,self.kwargs['max_sentence_length'])))
        self.register_parameter(name='not_target_logits_per_token', param=self.not_target_logits_per_token)
        
        self.reset()

    def reset(self):
        self.mm_ponderer.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
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
            feat_map = self.cnn_encoder(stin)
            # (mini_batch_size x though_space_depth_dim x visual_feat_spatial_dim x visual_feat_spatial_dim )
            features.append( feat_map)
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['though_space_depth_dim'], self.visual_feat_spatial_dim, self.visual_feat_spatial_dim)
        # (batch_size x nbr_distractors+1 / ? (descriptive mode depends on the role of the agent) x nbr_stimulus x though_space_depth_dim x ..nbr_visual_entity.. )

        features = self._makeVisualXYTSfeatures(features)
        # (batch_size x nbr_distractors+1 / ? (descriptive mode depends on the role of the agent) 
        # x nbr_stimulus x mm_ponderer_depth_dim=though_space_depth_dim+4 x ..nbr_visual_entity.. )

        return features 

    def _makeVisualXYTSfeatures(self, features):
        # (batch_size x nbr_distractors+1 x nbr_stimulus x though_space_depth_dim x ..nbr_visual_entity_per_stimulus.. )
        fsize = features.size()
        batch_size = fsize[0]
        nbr_distractors_po = fsize[1] 

        if self.visualXYDTS is None:
            sizeX = fsize[4]
            sizeY = fsize[5]
            sizeT = fsize[2]

            stepX = 2.0/sizeX
            stepY = 2.0/sizeY
            stepT = 1.0/sizeT

            fx = torch.zeros((1, 1, 1, 1, sizeX, 1))
            fy = torch.zeros((1, 1, 1, 1, 1, sizeY))
            ft = torch.zeros((1, 1, sizeT, 1, 1, 1))

            vx = -1+0.5*stepX
            for i in range(sizeX):
                fx[:,:,:,:,i,:] = vx 
                vx += stepX
            vy = -1+0.5*stepY
            for i in range(sizeY):
                fy[:,:,:,:,:,i] = vy 
                vy += stepY
            vt = 0.5*stepT
            for i in range(sizeT):
                ft[:,:,i,:,:,:] = vt
                vt += stepT

            fxyt = fx.repeat(1, 1, sizeT, 1, 1, sizeY)
            # (1 x 1 x nbr_stimulus x 1 x ..nbr_visual_entity.. )
            fyxt = fy.repeat(1, 1, sizeT, 1, sizeX, 1)
            # (1 x 1 x nbr_stimulus x 1 x ..nbr_visual_entity.. )
            ftxy = ft.repeat(1, 1, 1, 1, sizeX, sizeY)
            # (1 x 1 x nbr_stimulus x 1 x ..nbr_visual_entity.. )
            
            fsxyt = torch.ones((1, 1, sizeT, 1, sizeX, sizeY))

            fXYTS = torch.cat([fxyt, fyxt, ftxy, fsxyt], dim=3)
            self.visualXYTS = fXYTS
            # (1 x 1 x nbr_stimulus x 5 x ..nbr_visual_entity.. )
            
        fXYDTS = self.visualXYTS.repeat(batch_size, nbr_distractors_po, 1, 1, 1, 1)
        # (batch_size x nbr_distractors_po x nbr_stimulus x 5 x ..nbr_visual_entity_per_stimulus.. )

        if features.is_cuda: fXYDTS = fXYDTS.cuda()
        out = torch.cat([x, fXYDTS], dim=3)
        # (batch_size 
        # x nbr_distractors_po x nbr_stimulus 
        # x mm_ponderer_depth_dim=though_space_depth_dim+5 x ..nbr_visual_entity_per_stimulus.. )

        return out


    def _makeSymbolicXYTSfeatures(self, sentences):
        # (batch_size x sentence_length x though_space_depth_dim )
        # (batch_size, d=1, t=1, mm_ponderer_depth_dim=though_space_depth_dim+4, x*y=sentence_length)
        ssize = sentences.size()
        batch_size = ssize[0]
        sentence_length = ssize[1]

        sentences = sentences.transpose((0,2,1)).unsqueeze(1,2)
        # (batch_size x 1 x 1 x though_space_depth_dim x sentence_length)
        
        if self.symbolicXYDTS is None or sentence_length != self.sentence_length:
            self.sentence_length = sentence_length
            stepS = 2.0/self.sentence_length

            fx = torch.ones((1, 1, 1, 1, self.sentence_length))
            # (b=1 x d=1 x t=1 x 1 x ..nbr_symbolic_entity.. )
            fy = torch.ones((1, 1, 1, 1, self.sentence_length))
            # (b=1 x d=1 x t=1 x 1 x ..nbr_symbolic_entity.. )
            ft = torch.ones((1, 1, 1, 1, self.sentence_length))
            # (b=1 x d=1 x t=1 x 1 x ..nbr_symbolic_entity.. )
            
            fs = torch.zeros((1, 1, 1, 1, self.sentence_length))
            # (b=1 x d=1 x t=1 x 1 x ..nbr_symbolic_entity.. )
            vs = 0.5*stepS
            for i in range(self.sentence_length):
                fs[:,:,:,:,i] = vs
                vs += stepS

            sXYTS = torch.cat([fx, fy, ft, fs], dim=3)
            self.symbolicXYTS = sXYTS
            # (b=1 x d=1 x t=1 x 4 x ..nbr_symbolic_entity.. )
            
        sXYTS = self.symbolicXYTS.repeat(batch_size, 1, 1, 1, 1)
        # (batch_size x d=1 x t=1 x 4 x ..nbr_symbolic_entity.. )

        if sentences.is_cuda: sXYTS = sXYTS.cuda()
        out = torch.cat([sentences, sXYDTS], dim=3)
        # (batch_size x d=1 x t=1 x mm_ponderer_depth_dim=though_space_depth_dim+4 x ..nbr_symbolic_entity.. )
        return out
        
    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, nbr_distractors_po, nbr_stimulus, mm_ponderer_depth_dim=though_space_depth_dim+5, ..nbr_visual_entity..)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        batch_size = features.size(0)
        nbr_distractors_po = features.size(1)
        # (batch_size x nbr_distractors+1 / ? (descriptive mode depends on the role of the agent) x nbr_stimulus 
        # x mm_ponderer_depth_dim=though_space_depth_dim+5 x ..nbr_visual_entity.. )
        
        # Format the visual entities:
        self.visual_entities = features.view(*(features.size()[:-3]), -1).transpose(-2,-1).view((batch_size, nbr_distractors_po, -1, self.mm_ponderer_depth_dim))
        # (batch_size 
        # x nbr_distractors+1 / ? (descriptive mode depends on the role of the agent) 
        # x nbr_visual_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)
        
        # Format the symbolic entities:
        encoded_sentences = self.symbol_encoder( sentences.view((-1,self.vocab_size))).view(batch_size, -1, self.vocab_size)
        self.symbolic_entities = self._makeSymbolicXYTSfeatures(encoded_sentences)
        self.symbolic_entities = self.symbolic_entities.transpose(-2,-1).view((batch_size, 1, -1, self.mm_ponderer_depth_dim))
        # (batch_size 
        # x 1 * 1 * nbr_symbolic_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)
        self.symbolic_entities = self.symbolic_entities.repeat((1,nbr_distractors_po,1,1))
        # (batch_size 
        # x nbr_distractors_po
        # x nbr_symbolic_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)

        # Thoughts to ponder:
        self.thoughts = torch.cat([self.visual_entities, self.symbolic_entities], dim=1)
        # (batch_size 
        # x nbr_distractors_po
        # x nbr_visual_entity + nbr_symbolic_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)
        
        self.pondered_thoughts = []
        for d in range(nbr_distractors_po):
            self.pondered_thoughts.append( self.mm_ponderer(self.thoughts[:,d,...]).unsqueeze(1) )
        self.pondered_thoughts = torch.cat(self.pondered_thoughts, dim=1)
        # (batch_size 
        # x nbr_distractors_po
        # x nbr_visual_entity + nbr_symbolic_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)
        
        self.pondered_visual_entities = self.pondered_thoughts[:,:,:self.nbr_visual_entity,...]
        # (batch_size 
        # x nbr_distractors_po
        # x nbr_visual_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)
        
        self.pondered_symbolic_entities = self.pondered_thoughts[:,:,self.nbr_visual_entity+1:,...].mean(1)
        # (batch_size 
        # x nbr_symbolic_entity
        # x mm_ponderer_depth_dim=though_space_depth_dim+5)

        # Compute the decision: 
        decision_logits = []
        for widx in range(self.pondered_symbolic_entities.size(1)):
            decision_inputs = self.pondered_symbolic_entities[:,widx,...]
            # (batch_size, mm_ponderer_depth_dim)
            emb = self.pondered_visual_entities.view((batch_size, nbr_distractors_po, self.nbr_visual_entity, -1, 1)).transpose(-2,-1)
            #  batch_size, nbr_distractors_po, nbr_visual_entity, 1, mm_ponderer_depth_dim)
            din = decision_inputs[b].view((batch_size, 1, 1, -1, 1)).repeat(1, nbr_distractors_po, self.nbr_visual_entity, 1, 1)
            # (batch_size, nbr_distractors_po, nbr_visual_entity, mm_ponderer_depth_dim, 1)
            dl_per_visual_entity = torch.matmul( emb, din).squeeze()
            # (batch_size, nbr_distractors_po, nbr_visual_entity)
            decision_logits4widx = dl_per_visual_entity.mean(-1)
            # (batch_size, nbr_distractors_po)
            decision_logits.append(decision_logits4widx.unsqueeze(1))
            # (batch_size, 1, nbr_distractors_po)
        decision_logits = torch.cat(decision_logits, dim=1)
        # (batch_size, max_sentence_length, nbr_distractors_po)
        
        # Accumulate the decisions over the sentence in a forward manner:
        acc_idx = 0
        for idx in range(1,decision_logits.size(1)):
            decision_logits[:,idx,...] += (acc_idx+1)*decision_logits[:,acc_idx,:]
            acc_idx += 1
            decision_logits[:,idx,...] /= (acc_idx+1)

        if not(self.use_learning_not_target_logit):
            l_shape = decision_logits.size()
            not_target_logit = torch.zeros( *l_shape[:2], 1)
        else:
            not_target_logit = self.not_target_logits_per_token.view((1,self.max_sentence_length,1)).repeat(batch_size, 1, 1)
        if decision_logits.is_cuda: not_target_logit = not_target_logit.cuda()
        decision_logits = torch.cat([decision_logits, not_target_logit], dim=-1 )
        
        self.embedding_tf_final_outputs = self.pondered_visual_entities.view((batch_size, -1))
        
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
        # (batch_size x nbr_distractors+1 / ? (descriptive mode depends on the role of the agent) x nbr_stimulus 
        # x mm_ponderer_depth_dim=though_space_depth_dim+5 x ..nbr_visual_entity.. )
        
        # The operation (max/min) to use during the computation of the sentences
        # depends on the current role of the agent, that is determined by 
        # `sentences==None` ==> Speaker (first round).
        # TODO: account for multi-round communication...
        if sentences is None:   operation = torch.max 
        else:   operation = torch.min 

        # Similarly, as a speaker/teacher, it is assumed that `target_idx=0`.
        # TODO: decide on a strategy for the listener/student's predicted_target_idx's argument...
        predicted_target_idx = torch.zeros((batch_size, )).long()
        if visual_entities.is_cuda: predicted_target_idx = predicted_target_idx.cuda()

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
        next_sentences_one_hots = self._compute_sentence(features=features,
            target_idx=predicted_target_idx,
            _reason=self._reason,
            allowed_vocab_size=allowed_vocab_size,
            vocab_size=self.vocab_size,
            max_sentence_length=self.max_sentence_length,
            operation=operation,
            vocab_stop_idx=self.vocab_size-1,
            use_obverter_threshold_to_stop_message_generation=self.kwargs['use_obverter_threshold_to_stop_message_generation'],
            use_stop_word=False,
            _compute_tau=self._compute_tau,
            not_target_logits_per_token=self.not_target_logits_per_token if self.use_learning_not_target_logit else None,
            use_sentences_one_hot_vectors=self.use_sentences_one_hot_vectors,
            logger=self.logger)

        return next_sentences_widx, next_sentences_logits, next_sentences_one_hots, self.embedding_tf_final_outputs

    def _compute_sentence(self,
                          features, 
                          target_idx, 
                          _reason=None,
                          allowed_vocab_size=10, 
                          vocab_size=10, 
                          max_sentence_length=14,
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
        
        :param features: Tensor of (temporal) features embedding of shape `(batch_size, *self.obs_shape)`.
        :param target_idx: Tensor of indices of the target experiences of shape `(batch_size, 1)`.
        :param _reason: Function used to reason about the visual and textual entities.
        :param vocab_size: int, size of the vocabulary.
        :param max_sentence_length: int, maximal length for each generated sentences.
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
        import ipdb; ipdb.set_trace()

        batch_size = features.size(0)
        nbr_distractors_po = features.size(1)

        arange_vocab = torch.arange(vocab_size).float().view((1,vocab_size))
        arange_allowed_vocab = torch.arange(allowed_vocab_size).float()
        if features.is_cuda: 
            arange_vocab = arange_vocab.cuda()
            arange_allowed_vocab = arange_allowed_vocab.cuda()
        
        if use_sentences_one_hot_vectors:
            vocab_vectors = torch.zeros((allowed_vocab_size, vocab_size))
            # (allowed_vocab_size, vocab_size)
            for i in range(allowed_vocab_size): vocab_vectors[i,i] = 1.0
        else:
            vocab_vectors = torch.zeros((allowed_vocab_size,1)).long()
            # (allowed_vocab, 1)
            for i in range(allowed_vocab_size): vocab_vectors[i] = i
        if features.is_cuda: vocab_vectors = vocab_vectors.cuda()
        vocab_vectors = vocab_vectors.view((allowed_vocab_size,1,-1))
        # (batch_size=allowed_vocab_size, 1, vocab_size)

        sentences_widx = vocab_stop_idx*torch.ones((batch_size, max_sentence_length, 1))
        sentences_logits = torch.zeros((batch_size, max_sentence_length, vocab_size))
        sentences_one_hots =  torch.zeros((batch_size, max_sentence_length, vocab_size))
        sentences_logits[:,:,vocab_stop_idx] = 1.0
        sentences_one_hots[:,:,vocab_stop_idx] = 1.0

        if features.is_cuda: 
            sentences_widx = sentences_widx.cuda()
            sentences_logits = sentences_logits.cuda()
            sentences_one_hots = sentences_one_hots.cuda()


        relevant_batch = list(range(batch_size))
        for sentence_token_idx in range(max_sentence_length):
            sentences = sentences_one_hots[relevant_batch]
            # (cbatch_size, max_sentence_length, vocab_size)
            cbatch_size = sentences.size(0)

            sentences = sentences.unsqueeze(1).repeat(1, allowed_vocab_size, 1, 1) 
            # (cbatch_size, allowed_vocab_size, max_sentence_length, vocab_size)
            for vidx in range(allowed_vocab_size):
                sentences[:,vidx,sentence_token_idx,:] = vocab_vectors[vidx].view((1,-1)).repeat(cbatch_size,1)
            sentences = sentences.view((-1, max_sentence_length, vocab_size))
            # (cbatch_size*allowed_vocab_size, max_sentence_length, vocab_size)
            
            acc_decision_logits, _ = _reason(sentences, features)
            # (cbatch_size*allowed_vocab_size, max_sentence_length, nbr_distractors_po)
            decision_logits = acc_decision_logits[:,-1].view((cbatch_size, allowed_vocab_size*nbr_distractors_po))
            
            tau0 = 1e1
            # Probs over Distractors and Vocab: 
            decision_probs = F.softmax( decision_logits, dim=-1).view((cbatch_size, allowed_vocab_size, -1))
            decision_probs_least_effort = F.softmax( decision_logits*tau0, dim=-1).view((cbatch_size, allowed_vocab_size, -1))
            
            target_decision_probs_per_vocab_logits = decision_probs[:,:,btarget_idx]
            target_decision_probs_least_effort_per_vocab_logits = decision_probs_least_effort[:,:,btarget_idx]
            # (cbatch_size, allowed_vocab_size)
            
            tau = 1.0/5e0 
            if _compute_tau is not None:    
                cemb = features[relevant_batch]
                ctarget_idx = target_idx[relevant_batch]
                emb = []
                for cbidx, ctidx in enumerate(ctarget_idx):
                    emb.append(cemb[cbidx].index_select(index=ctidx, dim=0))
                emb = torch.cat(emb, dim=0)
                tau = _compute_tau(tau0=tau, emb=emb)
            else:
                tau = tau*torch.ones((cbatch_size,1))
            # The closer to zero this value is, the more accurate the operation is.
            if logger is not None: 
                it = 0
                key = "Obverter/ComputeSentenceTau"
                logger.add_scalar(key, tau.mean().item(), it)

            tau1 = 5e1
            # the higher this value is, the less random is the output.
            # and the more likely will it act like an argmax function...
            straight_through = True
            one_hot_sampled_vocab = gumbel_softmax(logits=target_decision_probs_per_vocab_logits*tau1, tau=tau, hard=straight_through, dim=-1)
            # (cbatch_size, allowed_vocab_size)

            if allowed_vocab_size < vocab_size:
                zeros4complete_vocab = torch.zeros((cbatch_size, vocab_size-allowed_vocab_size,))
                if one_hot_sampled_vocab.is_cuda: zeros4complete_vocab = zeros4complete_vocab.cuda()
                one_hot_sampled_vocab = torch.cat([one_hot_sampled_vocab, zeros4complete_vocab], dim=1)
                
                target_decision_probs_least_effort_per_vocab_logits = torch.cat([target_decision_probs_least_effort_per_vocab_logits,
                                                                                 zeros4complete_vocab], dim=1)

            vocab_idx_argop = torch.sum(arange_vocab.repeat(cbatch_size,1)*one_hot_sampled_vocab, dim=-1)
            # (cbatch_size)

            for (bidx, diff_widx_argop, td_logits) in zip(relevant_batch, vocab_idx_argop, target_decision_probs_least_effort_per_vocab_logits):
                vocab_idx_op = td_logits[diff_widx_argop.long()]
            
                sentences_widx[bidx, sentences_widx] = diff_widx_argop
                sentences_logits[bidx, sentences_widx] = td_logits.view((1,-1))
                sentences_one_hots[bidx, sentences_widx] = nn.functional.one_hot(diff_widx_argop.long(), num_classes=vocab_size).view((1,-1))
                
                if use_obverter_threshold_to_stop_message_generation:
                    if operation == torch.max:
                        operation_condition = (vocab_idx_op >= use_obverter_threshold_to_stop_message_generation)
                    else:
                        operation_condition = (vocab_idx_op < 1-use_obverter_threshold_to_stop_message_generation) 
                else:
                    operation_condition = False
                
                if use_stop_word:
                    stop_word_condition = (diff_widx_argop.long() == vocab_stop_idx)
                else:
                    stop_word_condition = False 

                if stop_word_condition or operation_condition:
                    relevant_batch.remove(bidx)

            if len(relevant_batch) == 0:    break


        return sentences_widx, sentences_logits, sentences_one_hots
