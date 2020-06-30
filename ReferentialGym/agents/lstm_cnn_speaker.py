import torch
import torch.nn as nn

from .rnn_cnn_speaker import RNNCNNSpeaker


class LSTMCNNSpeaker(RNNCNNSpeaker):
    def __init__(self, 
                    kwargs, 
                    obs_shape, 
                    vocab_size=100, 
                    max_sentence_length=10, 
                    agent_id='s0', 
                    logger=None):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=0` (partial observability), and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        '''
        super(LSTMCNNSpeaker, self).__init__(
            kwargs=kwargs,
            obs_shape=obs_shape, 
            vocab_size=vocab_size, 
            max_sentence_length=max_sentence_length, 
            agent_id=agent_id, 
            logger=logger, 
            rnn_type='lstm'
        )