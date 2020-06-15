import torch
import torch.nn as nn

from ..networks import layer_init
from ..utils import gumbel_softmax
from .agent import Agent 


def sentence_length_logging_hook(agent,
                                 losses_dict,
                                 input_streams_dict,
                                 outputs_dict,
                                 logs_dict,
                                 **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    mode = input_streams_dict["mode"]
    config = input_streams_dict["config"]

    batch_size = len(input_streams_dict["experiences"])
    speaker_sentences_logits = outputs_dict["sentences_logits"]
    speaker_sentences_widx = outputs_dict["sentences_widx"]

    # Sentence Lengths:
    """
    #if "obverter" in input_streams_dict["graphtype"].lower():
    sentence_lengths = torch.sum(-(speaker_sentences_widx.squeeze(-1)-agent.vocab_size).sign(), dim=-1).reshape(batch_size,-1)
    sentence_length = sentence_lengths.mean()
    """
    sentence_lengths = (speaker_sentences_widx < (agent.vocab_pad_idx))
    sentence_lengths = sentence_lengths.reshape(batch_size,-1).float().sum(-1)
    sentence_length = sentence_lengths.mean()
    
    logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{agent.agent_id}/SentenceLength (/{config['max_sentence_length']})"] = sentence_lengths/config["max_sentence_length"]


def entropy_logging_hook(agent,
                         losses_dict,
                         input_streams_dict,
                         outputs_dict,
                         logs_dict,
                         **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    mode = input_streams_dict["mode"]
    config = input_streams_dict["config"]

    batch_size = len(input_streams_dict["experiences"])
    speaker_sentences_logits = outputs_dict["sentences_logits"]
    speaker_sentences_widx = outputs_dict["sentences_widx"]
    
    # Compute Sentence Entropies:
    sentences_log_probs = [
        s_logits.reshape(-1,agent.vocab_size).log_softmax(dim=-1) 
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


def entropy_regularization_loss_hook(agent,
                                     losses_dict,
                                     input_streams_dict,
                                     outputs_dict,
                                     logs_dict,
                                     **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]

    entropies_per_sentence = torch.cat(
        [ torch.cat(
            [ torch.distributions.categorical.Categorical(logits=w_logits).entropy().view(1,1) 
              for w_logits in s_logits
            ], 
            dim=-1).mean(dim=-1) 
          for s_logits in outputs_dict["sentences_logits"]
        ], 
        dim=0
    )
    # (batch_size, 1)
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/speaker_entropy_regularization_loss"] = [
        config["entropy_regularization_factor"], 
        entropies_per_sentence.squeeze()
    ]
    # (batch_size, )

    """
    # Entropy minimization:
    distr = torch.distributions.Categorical(probs=decision_probs)
    entropy_loss = distr.entropy()
    losses_dict["entropy_loss"] = [1.0, entropy_loss]
    """


def mdl_principle_loss_hook(agent,
                            losses_dict,
                            input_streams_dict,
                            outputs_dict,
                            logs_dict,
                            **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]

    batch_size = len(input_streams_dict["experiences"])

    arange_token = torch.arange(config["max_sentence_length"])
    arange_token = (config["vocab_size"]*arange_token).float().view((1,-1)).repeat(batch_size,1)
    if config["use_cuda"]: arange_token = arange_token.cuda()
    non_pad_mask = (outputs_dict["sentences_widx"] < (agent.vocab_size)).float()
    # (batch_size, max_sentence_length, 1)
    if config["use_cuda"]: mask = mask.cuda()
    speaker_reweighted_utterances = 1+non_pad_mask*outputs_dict["sentences_widx"] \
        -(1-non_pad_mask)*outputs_dict["sentences_widx"]/config["vocab_size"]
    mdl_loss = (arange_token+speaker_reweighted_utterances.squeeze()).mean(dim=-1)
    # (batch_size, )
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/mdl_loss"] = [
        config["mdl_principle_factor"], 
        mdl_loss
    ]   


def oov_loss_hook(agent,
                  losses_dict,
                  input_streams_dict,
                  outputs_dict,
                  logs_dict,
                  **kwargs):
    it_rep = input_streams_dict["it_rep"]
    it_comm_round = input_streams_dict["it_comm_round"]
    config = input_streams_dict["config"]

    batch_size = len(input_streams_dict["experiences"])

    arange_vocab = torch.arange(config["vocab_size"]+1).float()
    if config["use_cuda"]: arange_vocab = arange_vocab.cuda()
    speaker_utterances = torch.cat(
        [((s+1) / (s.detach()+1)) * torch.nn.functional.one_hot(s.long().squeeze(), num_classes=config["vocab_size"]+1).float().unsqueeze(0)
        for s in outputs_dict["sentences_widx"]], 
        dim=0
    )
    # (batch_size, sentence_length,vocab_size+1)
    speaker_utterances_count = speaker_utterances.sum(dim=0).sum(dim=0).float().squeeze()
    outputs_dict["speaker_utterances_count"] = speaker_utterances_count
    # (vocab_size+1,)
    total_nbr_utterances = speaker_utterances_count.sum().item()
    d_speaker_utterances_probs = (speaker_utterances_count/(config["utterance_oov_prob"]+total_nbr_utterances-1)).detach()
    # (vocab_size+1,)
    #oov_loss = -(1.0/(batch_size*config["max_sentence_length"]))*torch.sum(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
    oov_loss = -(1.0/(batch_size*config["max_sentence_length"]))*(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
    # (batch_size, 1)
    if config["with_utterance_promotion"]:
        oov_loss *= -1 
    losses_dict[f"repetition{it_rep}/comm_round{it_comm_round}/oov_loss"] = [config["utterance_factor"], oov_loss]


class Speaker(Agent):
    def __init__(self, obs_shape, vocab_size=100, max_sentence_length=10, agent_id="s0", logger=None, kwargs=None):
        """
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        """
        super(Speaker, self).__init__(agent_id=agent_id, 
                                      obs_shape=obs_shape,
                                      vocab_size=vocab_size,
                                      max_sentence_length=max_sentence_length,
                                      logger=logger, 
                                      kwargs=kwargs,
                                      role="speaker")
        
        
        self.register_hook(sentence_length_logging_hook)
        self.register_hook(entropy_logging_hook)

        if "with_speaker_entropy_regularization" in self.kwargs \
         and self.kwargs["with_speaker_entropy_regularization"]:
            self.register_hook(entropy_regularization_loss_hook)

        if "with_mdl_principle" in self.kwargs \
         and self.kwargs["with_mdl_principle"]:
            self.register_hook(mdl_principle_loss_hook)

        if ("with_utterance_penalization" in self.kwargs or "with_utterance_promotion" in self.kwargs) \
         and (self.kwargs["with_utterance_penalization"] or self.kwargs["with_utterance_promotion"]):
            self.register_hook(oov_loss_hook)

        # Multi-round:
        self._reset_rnn_states()

    def reset(self):
        self.apply(layer_init)

    def _reset_rnn_states(self):
        self.rnn_states = None

    def _compute_tau(self, tau0):
        raise NotImplementedError

    def _sense(self, experiences, sentences=None):
        """
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `experiences[:, 0]` is assumed as the target experience, while the others are distractors, if any. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        """

        raise NotImplementedError

    def _utter(self, features, sentences=None):
        """
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        raise NotImplementedError

    def forward(self, experiences, sentences=None, multi_round=False, graphtype="straight_through_gumbel_softmax", tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                            `experiences[:,0]` is assumed as the target experience, while the others are distractors, if any. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        batch_size = experiences.size(0)
        features = self._sense(experiences=experiences, sentences=sentences)
        utter_outputs = self._utter(features=features, sentences=sentences)
        if len(utter_outputs) == 5:
            next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        else:
            next_sentences_hidden_states = None
            next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        
        if self.training:
            if "gumbel_softmax" in graphtype:    
                if next_sentences_hidden_states is None: 
                    self.tau = self._compute_tau(tau0=tau0)
                    #tau = self.tau.view((-1,1,1)).repeat(1, self.max_sentence_length, self.vocab_size)
                    tau = self.tau.view((-1))
                    # (batch_size)
                else:
                    self.tau = []
                    for hs in next_sentences_hidden_states:
                        self.tau.append( self._compute_tau(tau0=tau0, h=hs).view((-1)))
                    # list of size batch_size containing Tensors of shape (sentence_length)
                    tau = self.tau 
                    
                straight_through = (graphtype == "straight_through_gumbel_softmax")
                
                next_sentences_stgs = []
                for bidx in range(len(next_sentences_logits)):
                    nsl_in = next_sentences_logits[bidx]
                    # (sentence_length<=max_sentence_length, vocab_size)
                    tau_in = tau[bidx].view((-1,1))
                    # (1, 1) or (sentence_length, 1)
                    stgs = gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1, eps=self.kwargs["gumbel_softmax_eps"])
                    
                    next_sentences_stgs.append(stgs)
                    #next_sentences_stgs.append( nn.functional.gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                next_sentences = next_sentences_stgs
                if isinstance(next_sentences, list): 
                    next_sentences = nn.utils.rnn.pad_sequence(next_sentences, batch_first=True, padding_value=0.0).float()
                    # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)

        output_dict = {"sentences_widx":next_sentences_widx, 
                       "sentences_logits":next_sentences_logits, 
                       "sentences_one_hot":next_sentences,
                       #"features":features,
                       "temporal_features":temporal_features}
        
        if not multi_round:
            self._reset_rnn_states()

        return output_dict