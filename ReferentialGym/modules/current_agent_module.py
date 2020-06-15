from typing import Dict, List 

from ..modules import Module


class CurrentAgentModule(Module):
    def __init__(self, 
                 id="current_agent", 
                 role=None):
        """
        :param id: str defining the ID of the module.
        :param role: str defining the role of the agent, e.g. "speaker"/"listener".
        """

        super(CurrentAgentModule, self).__init__(id=id, 
                                                type="CurrentAgentModule",
                                                config=None,
                                                input_stream_ids=None)

        self.ref_agent = None 
        self.role = role

    def set_ref(self, agent):
        self.ref_agent = agent 

    def get_input_stream_ids(self):
        return self.ref_agent.get_input_stream_ids()

    def clone(self, clone_id="a0"):
        return self.ref_agent.clone()

    def save(self, path):
        self.ref_agent.save(path=path)

    def _tidyup(self):
        self.ref_agent._tidyup()

    def _log(self, log_dict, batch_size):
        self.ref_agent._log()

    def register_hook(self, hook):
        self.ref_agent.register_hook(hook=hook)

    def forward(self, sentences, experiences, multi_round=False, graphtype="straight_through_gumbel_softmax", tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        self.ref_agent.role = self.role
        return self.ref_agent(sentences=sentences, experiences=experiences, multi_round=multi_round, graphtype=graphtype, tau0=tau0)


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
        self.ref_agent.role = self.role
        outputs_dict = self.ref_agent.compute(input_streams_dict=input_streams_dict)

        outputs_dict["ref_agent"] = self.ref_agent

        return outputs_dict