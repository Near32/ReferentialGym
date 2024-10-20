from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module


def build_InstantaneousCoordinationMetricModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return InstantaneousCoordinationMetricModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class InstantaneousCoordinationMetricModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",

            "end_of_dataset":"signals:end_of_dataset",  
            # boolean: whether the current batch/datasample is the last of the current dataset/mode.
            "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
            # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
            # is the last of the current sequence of repetition.
            "end_of_communication":"signals:end_of_communication",
            # boolean: whether the current communication round is the last of 
            # the current dialog.
            "dataset":"current_dataset:ref",

            "vocab_size":"config:vocab_size",
            "max_sentence_length":"config:max_sentence_length",
            "sentences_widx":"modules:current_speaker:sentences_widx", 
            "decision_probs":"modules:current_listener:decision_probs",
            "listener_indices":"current_dataloader:sample:listener_indices",
        }
        
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids:
                    input_stream_ids[default_id] = default_stream

        super(InstantaneousCoordinationMetricModule, self).__init__(id=id,
                                                 type="InstantaneousCoordinationMetricModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.sentences_widx = []
        self.decision_probs = []
        self.listener_indices = []

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if epoch % self.config["epoch_period"] == 0:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                sentences_widx = input_streams_dict["sentences_widx"]
                self.sentences_widx.append(sentences_widx.cpu().detach().long().numpy())
                # (1, batch_size, max_sentence_length, 1) 
                decision_probs = input_streams_dict["decision_probs"]
                self.decision_probs.append(decision_probs.cpu().detach().numpy())
                # (1, batch_size, nbr_stimulus) 
                listener_indices = input_streams_dict["listener_indices"]
                self.listener_indices.append(listener_indices.cpu().detach().long().numpy())
                # (1, batch_size, nbr_stimulus) 

            # Is it the end of the epoch?
            end_of_epoch = all([
              input_streams_dict[key]
              for key in self.end_of_]
            )
            
            not_empty = len(self.decision_probs) > 0

            if end_of_epoch and not_empty:
                # self.sentences_widx = np.concatenate(self.sentences_widx, axis=0)
                # (nbr_element, batch_size may vary.., nbr_stimulus) 
                #self.decision_probs = np.concatenate(self.decision_probs, axis=0)
                # (nbr_element, batch_size may vary.., nbr_stimulus) 
                #self.listener_indices = np.concatenate(self.listener_indices, axis=0)
                # (nbr_element, batch_size may vary.., nbr_stimulus)
                # Account for descriptive mode:
                non_target_stimulus_idx = (-1)*np.ones((1, 1))
                self.listener_indices = [
                    np.concatenate([np.reshape(el, (el.shape[0], -1)), np.tile(non_target_stimulus_idx, reps=(el.shape[0], 1))], axis=-1)
                    for el in self.listener_indices
                ]
                # (nbr_element, batch_size may vary.., nbr_stimulus+1)
                
                nbr_element =  len(self.decision_probs)
                nbr_possible_listener_actions = self.decision_probs[0].shape[-1]
                nbr_possible_unique_sentences = input_streams_dict["vocab_size"]**input_streams_dict["max_sentence_length"]
                
                self.listener_decision_stimulus_indices = [
                    #self.decision_probs[el].max(axis=-1).reshape(-1, 1) 
                    self.decision_probs[el].argmax(axis=-1).reshape(-1, 1) 
                    for el in range(nbr_element)
                ]
                # (nbr_element, batch_size, 1)
                self.listener_decision_indices = [
                    np.take_along_axis(self.listener_indices[el], self.listener_decision_stimulus_indices[el].astype(int), axis=-1)
                    for el in range(nbr_element)
                ]
                # (nbr_element, batch_size, 1) 
                
                logger = input_streams_dict["logger"]

                co_occ_matrix = {}
                nbr_decisions = 0
                for batch_listener_actions, batch_sentences in zip(self.listener_decision_indices, self.sentences_widx):
                    for batch_idx in range(batch_listener_actions.shape[0]):
                        action = int(batch_listener_actions[batch_idx].item())
                        sentence = batch_sentences[batch_idx].tobytes()
                        if sentence not in co_occ_matrix:
                            co_occ_matrix[sentence] = {} 
                        if action not in co_occ_matrix[sentence]:
                            co_occ_matrix[sentence][action] = 0

                        co_occ_matrix[sentence][action] += 1
                        nbr_decisions +=1


                IC = 0
                marg_p = {}
                for sentence in co_occ_matrix:
                    marg_p[sentence] = sum(
                        [
                            p_sentence_action 
                            for p_sentence_action in co_occ_matrix[sentence].values()
                        ]
                    )
                    for action in co_occ_matrix[sentence]:
                        if action not in marg_p:
                            marg_p[action] = sum(
                                [
                                    joint_p[action] if action in joint_p else 0
                                    for joint_p in co_occ_matrix.values()
                                ]
                            )
                        joint_p_sentence_action = co_occ_matrix[sentence][action]/nbr_decisions
                        IC += joint_p_sentence_action*np.log2(joint_p_sentence_action/(marg_p[action]*marg_p[sentence]))

                logs_dict[f"{mode}/{self.id}/IC"] = IC
                
                self.sentences_widx = []  
                self.decision_probs = []
                self.listener_indices = []
            
        return outputs_stream_dict
    
