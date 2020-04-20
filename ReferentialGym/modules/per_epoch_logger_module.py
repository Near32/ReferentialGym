from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module

def build_PerEpochLoggerModule(id:str,
                               config:Dict[str,object]=None,
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return PerEpochLoggerModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class PerEpochLoggerModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
                "modules:logger:ref":"logger",
                "losses_dict":"losses_dict",
                "logs_dict":"logs_dict",
                "signals:epoch":"epoch",
                "signals:mode":"mode",
                "signals:end_of_dataset":"end_of_dataset",  
                # boolean: whether the current batch/datasample is the last of the current dataset/mode.
                "signals:it_datasample":"global_it_datasample",
                "signals:it_datasample":"it_datasample",
                "signals:end_of_repetition_sequence":"end_of_repetition_sequence",
                # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
                # is the last of the current sequence of repetition.
                "signals:global_it_sample":"global_it_sample",
                "signals:it_sample":"it_sample",
                # step in the sequence of repetitions of the current batch
                "signals:end_of_communication":"end_of_communication",
                # boolean: whether the current communication round is the last of 
                # the current dialog.
                "signals:global_it_step":"global_it_step",
                "signals:it_step":"it_step",
                # step in the communication round.
            }

        assert "logger" in input_stream_ids.values(),\
               "PerEpochLoggerModule relies on 'logger' id.\n\
                Not found in input_stream_ids."
        
        assert "epoch" in input_stream_ids.values(),\
               "PerEpochLoggerModule relies on 'epoch' id.\n\
                Not found in input_stream_ids."
        
        assert "mode" in input_stream_ids.values(),\
               "PerEpochLoggerModule relies on 'mode' id.\n\
                Not found in input_stream_ids."
        
        assert "losses_dict" in input_stream_ids.values(),\
               "PerEpochLoggerModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids."

        assert "logs_dict" in input_stream_ids.values(),\
               "PerEpochLoggerModule relies on 'logs_dict' id.\n\
                Not found in input_stream_ids."
        
        super(PerEpochLoggerModule, self).__init__(id=id,
                                                 type="PerEpochLoggerModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.storages = {}

        self.end_of_ = [value for key,value in input_stream_ids.items() if 'end_of_' in value]
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        '''
        '''
        outputs_stream_dict = {}

        losses_dict = input_streams_dict['losses_dict']
        logs_dict = input_streams_dict['logs_dict']
        
        epoch = input_streams_dict['epoch']
        mode = input_streams_dict['mode']
        global_it_step = input_streams_dict['global_it_step']
        
        logger = input_streams_dict['logger']

        # Store new data:
        for key,value in logs_dict.items():
          if key not in self.storages:
            self.storages[key] = []
          self.storages[key].append(value)
        
        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        # If so, let us average over every value and save it:
        if end_of_epoch:
          for key, valuelist in self.storages.items():
            need_mean_std = False
            if isinstance(valuelist[0], torch.Tensor) and len(valuelist[0].shape)>=1:
              values = torch.stack(valuelist).cpu().detach().reshape(-1).numpy()
              need_mean_std = True
              averaged_value = values.mean()
              std_value = values.std()
            elif isinstance(valuelist[0], float):
              values = np.asarray(valuelist).reshape(-1)
              if len(valuelist)>1:
                need_mean_std = True
              averaged_value = values.mean()
              std_value = values.std()
            else:
              continue

            if need_mean_std:
              logger.add_scalar(f"PerEpoch/{key}/Mean", averaged_value, epoch)
              logger.add_scalar(f"PerEpoch/{key}/Std", std_value, epoch)
              logger.add_histogram(f"PerEpoch/{key}", values, epoch)
            else:
              logger.add_scalar(f"PerEpoch/{key}", averaged_value, epoch)
              # Remove the value form the logs_dict if it is present:
              logs_dict.pop(key, None)

          # Reset epoch storages:
          self.storages = {}

          # Flush data:
          logger.flush()


        # Log new (rectified) data:
        for key,value in logs_dict.items():
          if isinstance(value, torch.Tensor): 
              value = value.mean().item()
          logger.add_scalar(key, value, global_it_step)

        return {}
        