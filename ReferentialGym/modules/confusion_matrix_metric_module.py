from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
from sklearn.metrics import precision_recall_fscore_support

from .module import Module

def build_ConfusionMatrixMetricModule(id:str, config:Dict[str,object], input_stream_ids:Dict[str,str]) -> Module:
    return ConfusionMatrixMetricModule(id=id, config=config, input_stream_ids=input_stream_ids)


class ConfusionMatrixMetricModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]):

        input_stream_ids.update(
          {
              "logger":"modules:logger:ref",
              "logs_dict":"logs_dict",
              "epoch":"signals:epoch",
              "mode":"signals:mode",
              "end_of_dataset":"signals:end_of_dataset",  
              # boolean: whether the current batch/datasample is the last of the current dataset/mode.
              "global_it_datasample":"signals:it_datasample",
              "it_datasample":"signals:it_datasample",
              "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
              # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
              # is the last of the current sequence of repetition.
              "global_it_sample":"signals:global_it_sample",
              "it_sample":"signals:it_sample",
              # step in the sequence of repetitions of the current batch
              "end_of_communication":"signals:end_of_communication",
              # boolean: whether the current communication round is the last of 
              # the current dialog.
              "global_it_step":"signals:global_it_step",
              "it_step":"signals:it_step",
              # step in the communication round.
          }
        )

        assert "predicted_labels_0" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on, at least, 'predicted_labels_0' id to start its pipeline.\n\
                Not found in input_stream_ids."
        assert "groundtruth_labels_0" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on, at least, 'groundtruth_labels_0' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        
        assert "logger" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on 'logger' id.\n\
                Not found in input_stream_ids."
        
        assert "epoch" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on 'epoch' id.\n\
                Not found in input_stream_ids."
        
        assert "mode" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on 'mode' id.\n\
                Not found in input_stream_ids."
        
        assert "logs_dict" in input_stream_ids.keys(),\
               "ConfusionMatrixMetricModule relies on 'logs_dict' id.\n\
                Not found in input_stream_ids."
        
        super(ConfusionMatrixMetricModule, self).__init__(id=id,
                                                 type="ConfusionMatrixMetricModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.prediction_storages = {}
        self.groundtruth_storages = {}

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        logs_dict = input_streams_dict["logs_dict"]
        
        epoch = input_streams_dict["epoch"]
        mode = input_streams_dict["mode"]
        global_it_step = input_streams_dict["global_it_step"]
        
        logger = input_streams_dict["logger"]

        predicted_labels = {k:v for k,v in input_streams_dict.items() if "predicted_labels" in k}
        groundtruth_labels = {k:v for k,v in input_streams_dict.items() if "groundtruth_labels" in k}
        
        
        # Store new data:
        for key, value in predicted_labels.items():
          if key not in self.prediction_storages:
            self.prediction_storages[key] = {}
          for kp, vp in value.items():
            if kp not in self.prediction_storages[key]:
              self.prediction_storages[key][kp] = []
            self.prediction_storages[key][kp].append(vp)


        for key, value in groundtruth_labels.items():
          if key not in self.groundtruth_storages:
            self.groundtruth_storages[key] = {}
          for kp, vp in value.items():
            if kp not in self.groundtruth_storages[key]:
              self.groundtruth_storages[key][kp] = []
            self.groundtruth_storages[key][kp].append(vp)
        
        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        # If so, let us average over every value and save it:
        if end_of_epoch:

          pc_accuracies= {}
          precisions = {}
          recalls = {}
          f_scores = {}
          for pkey_idx, key_input in enumerate(self.prediction_storages.keys()):
            pc_accuracies[key_input] = {}
            precisions[key_input] = {}
            recalls[key_input] = {}
            f_scores[key_input] = {}
            
            for key in self.prediction_storages[key_input].keys():
              # Stack:
              pv = np.concatenate([el.ravel() for el in self.prediction_storages[key_input][key]])
              gtv = np.concatenate([el.ravel() for el in self.groundtruth_storages[f"groundtruth_labels_{pkey_idx}"][key]])

              # Compute Precision, Recall, F-Score, 
              # with inversely proportional weighting to account for imbalanced support:
              has_support_idx = set(gtv)
              sample_weights = np.zeros_like(gtv, dtype=float)
              for sidx in has_support_idx:
                mask = (gtv == sidx).astype(float)
                support_size = mask.sum()
                sample_weights += (1.0/support_size)*mask

              precisions[key_input][f"balanced_{key}"], \
              recalls[key_input][f"balanced_{key}"], \
              f_scores[key_input][f"balanced_{key}"], _ = precision_recall_fscore_support(
                y_true=gtv, 
                y_pred=pv, 
                labels=self.config["labels"],
                average='macro',
                sample_weight=sample_weights,
                zero_division=0,
              )
              # (1)
              
              # Compute Precision, Recall, F-Score, for each class:
              precisions[key_input][key], \
              recalls[key_input][key], \
              f_scores[key_input][key], _ = precision_recall_fscore_support(
                y_true=gtv, 
                y_pred=pv, 
                labels=self.config["labels"],
                average=None,
                zero_division=0,
              )
              # (nb_class)
              
              # Eliminate the classes without support:
              
              temp = precisions[key_input][key]
              precisions[key_input][key] = {}
              for sidx in has_support_idx:
                precisions[key_input][key][sidx] = temp[sidx]

              temp = recalls[key_input][key]
              recalls[key_input][key] = {}
              for sidx in has_support_idx:
                recalls[key_input][key][sidx] = temp[sidx]

              temp = f_scores[key_input][key]
              f_scores[key_input][key] = {}
              for sidx in has_support_idx:
                f_scores[key_input][key][sidx] = temp[sidx]

              # Compute Per-Class Accuracies where there is support:
              pc_accuracies[key_input][key] = {}
              for sidx in has_support_idx:
                class_mask = (gtv == sidx)
                pc_accuracies[key_input][key][sidx] = 100.0*class_mask*(pv==gtv).astype(float)

              # Logging:
              logger.add_scalar(
                f"PerEpoch/{mode}/{self.id}/Precision/{self.config['input_labels'][key_input]}/{key}/balanced", 
                precisions[key_input][f"balanced_{key}"], 
                epoch
              )

              logger.add_scalar(
                f"PerEpoch/{mode}/{self.id}/Recall/{self.config['input_labels'][key_input]}/{key}/balanced", 
                recalls[key_input][f"balanced_{key}"], 
                epoch
              )

              logger.add_scalar(
                f"PerEpoch/{mode}/{self.id}/FScore/{self.config['input_labels'][key_input]}/{key}/balanced", 
                f_scores[key_input][f"balanced_{key}"], 
                epoch
              )
              for sidx in has_support_idx:
                logger.add_scalar(
                  f"PerEpoch/{mode}/{self.id}/Precision/{self.config['input_labels'][key_input]}/{key}/class_{sidx}", 
                  precisions[key_input][key][sidx], 
                  epoch
                )

                logger.add_scalar(
                  f"PerEpoch/{mode}/{self.id}/Recall/{self.config['input_labels'][key_input]}/{key}/class_{sidx}", 
                  recalls[key_input][key][sidx], 
                  epoch
                )

                logger.add_scalar(
                  f"PerEpoch/{mode}/{self.id}/FScore/{self.config['input_labels'][key_input]}/{key}/class_{sidx}", 
                  f_scores[key_input][key][sidx], 
                  epoch
                )

                logger.add_scalar(
                  f"PerEpoch/{mode}/{self.id}/PerClassAccuracy/{self.config['input_labels'][key_input]}/{key}/class_{sidx}", 
                  pc_accuracies[key_input][key][sidx].mean(), 
                  epoch
                )
              
          # Reset epoch storages:
          self.prediction_storages = {}
          self.groundtruth_storages = {}

          # Flush data:
          logger.flush()

        return {}
        