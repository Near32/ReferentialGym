from typing import Dict, List 

import torch
import numpy as np
import copy 

import wandb 

import ReferentialGym
from ReferentialGym.modules import Module


class OCCModule(Module):
    def __init__(
        self, 
        id:str,
        config:Dict[str,object]
        ):
        """
        :param id: str defining the ID of the module.
        :param config: Dict of parameters, expecting:
            - 
        """

        input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch_logs_dict":"modules:per_epoch_logger:ref:latest_logs",
            "epoch":"signals:epoch",
            "mode":"signals:mode",
            
            "it_step":"signals:it_step",
            # step in the communication round.
         
            "sample":"current_dataloader:sample",

            "end_of_dataset":"signals:end_of_dataset",  
            # boolean: whether the current batch/datasample is the last of the current dataset/mode.
            "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
            # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
            # is the last of the current sequence of repetition.
            "end_of_communication":"signals:end_of_communication",
            # boolean: whether the current communication round is the last of 
            # the current dialog.
            "dataset":"current_dataset:ref",
        }

        super(OCCModule, self).__init__(
            id=id, 
            type="OCCModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        self.accuracy_threshold = self.config.get('accuracy_threshold', 10.0)
        self.hard_object_centric_ratio = 0.0
        
    def update_object_centric_type(self, input_streams_dict:Dict[str,object]):
        dataset = input_streams_dict['dataset']
        assert dataset.kwargs['object_centric']
        dataset.original_object_centric_type = f"ratio-{self.hard_object_centric_ratio}"

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        
        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
        """

        outputs_dict = {}

        logs_dict = input_streams_dict["logs_dict"]
        epoch_logs_dict = input_streams_dict["epoch_logs_dict"]
        epoch = input_streams_dict["epoch"]
        mode = input_streams_dict["mode"]
        it_step = input_streams_dict["it_step"]
        
        dataset = input_streams_dict["dataset"]
        dataset_size = dataset.size()
        
        if mode=='test' \
        and it_step == 0:
            # Is it the end of the epoch?
            end_of_epoch = all([input_streams_dict[key] for key in self.end_of_])
            
            if end_of_epoch:
                test_accuracy = epoch_logs_dict.get(
                    "PerEpoch/test/repetition0/comm_round0/referential_game_accuracy/Mean",
                    0.0,
                )
                if test_accuracy >= self.accuracy_threshold:
                    self.hard_object_centric_ratio = int(100*(test_accuracy-self.accuracy_threshold)/(100-self.accuracy_threshold))
                else:
                    self.hard_object_centric_ratio = 0
                
                wandb.log({
                    "ObjectCentricCurriculum/OCC_ratio": self.hard_object_centric_ratio,
                    "ObjectCentricCurriculum/TestAccuracy": test_accuracy,
                    "ObjectCentricCurriculum/AccuracyThreshold": self.accuracy_threshold,
                    },
                    commit=False,
                )
        
        self.update_object_centric_type(input_streams_dict)
        
        return outputs_dict

   
