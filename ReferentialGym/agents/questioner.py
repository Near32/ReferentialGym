import torch
import torch.nn as nn

from ..networks import layer_init
from ..utils import gumbel_softmax
from .conversational_agent import ConversationalAgent 


class Questioner(ConversationalAgent):
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
        super(Questioner, self).__init__(
            agent_id=agent_id, 
            obs_shape=obs_shape,
            vocab_size=vocab_size,
            max_sentence_length=max_sentence_length,
            logger=logger, 
            kwargs=kwargs,
            role="questioner")
        
        self.inner_listener = None 

    def _register_inner_listener(self, inner_listener):
        self.inner_listener = inner_listener
        for hook in self.inner_listener.hooks:
            self.register_hook(hook)

    def _compute_tau(self, tau0):
        raise NotImplementedError

    def _tidyup(self):
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

    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
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

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Compute the losses and return them along with the produced outputs.

        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_logits'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits over symbols.
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sentences_one_hot'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
            - `'exp_latents'`: Tensor of shape `(batch_size, nbr_latent_dimensions)`.
            - `'multi_round'`: Boolean defining whether to utter a sentence back or not.
            - `'graphtype'`: String defining the type of symbols used in the output sentence:
                        - `'categorical'`: one-hot-encoded symbols.
                        - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                        - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                        - `'obverter'`: obverter training scheme...
            - `'tau0'`: Float, temperature with which to apply gumbel-softmax estimator. 
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'config'`: Dict of hyperparameters to the referential game.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
            - `'it'`: Integer specifying the iteration number of the current function call.
        """
        config = input_streams_dict["config"]
        mode = input_streams_dict["mode"]
        it_rep = input_streams_dict["it_rep"]
        it_comm_round = input_streams_dict["it_comm_round"]
        global_it_comm_round = input_streams_dict["global_it_comm_round"]
        
        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]
        
        input_sentence = input_streams_dict["sentences_widx"]
        if self.use_sentences_one_hot_vectors:
            input_sentence = input_streams_dict["sentences_one_hot"]

        if input_streams_dict["experiences"] is not None:
            batch_size = input_streams_dict["experiences"].shape[0]
        else:
            batch_size = input_sentence.shape[0]
            
        outputs_dict = self(sentences=input_sentence,
                           experiences=input_streams_dict["experiences"],
                           multi_round=input_streams_dict["multi_round"],
                           graphtype=input_streams_dict["graphtype"],
                           tau0=input_streams_dict["tau0"])

        ## Accounts for the fact that it is a discriminative agent:
        outputs_dict["decision"] = outputs_dict["output"]

        if "exp_latents" in input_streams_dict:
            outputs_dict["exp_latents"] = input_streams_dict["exp_latents"]
        if "exp_latents_values" in input_streams_dict:
            outputs_dict["exp_latents_values"] = input_streams_dict["exp_latents_values"]
        
        self._log(outputs_dict, batch_size=batch_size)

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//

        """
        if hasattr(self, "TC_losses"):
            losses_dict[f"{self.role}/TC_loss"] = [1.0, self.TC_losses]
        """
        if hasattr(self, "VAE_losses") and vae_loss_hook not in self.hooks:
            self.register_hook(vae_loss_hook)

        if hasattr(self,"tau"): 
            tau = torch.cat([ t.view((-1)) for t in self.tau], dim=0) if isinstance(self.tau, list) else self.tau
            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/Tau/{self.agent_id}"] = tau
        
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        
        for hook in self.hooks:
            hook(
                agent=self,
                losses_dict=losses_dict,
                input_streams_dict=input_streams_dict,
                outputs_dict=outputs_dict,
                logs_dict=logs_dict
            )

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        
        # Logging:        
        for logname, value in self.log_dict.items():
            self.logger.add_scalar(f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/{self.role}/{logname}", value.item(), global_it_comm_round)
        self.log_dict = {}

        self._tidyup()
        
        outputs_dict["losses"] = losses_dict

        return outputs_dict    
