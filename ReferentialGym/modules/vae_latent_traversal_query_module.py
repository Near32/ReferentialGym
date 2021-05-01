from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module
from ReferentialGym.utils import query_vae_latent_space

def build_VAELatentTraversalQueryModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return VAELatentTraversalQueryModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class VAELatentTraversalQueryModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None):

        default_input_stream_ids = {
            "logger":"modules:logger:ref",
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

            "model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
            "experiences":"current_dataloader:sample:speaker_experiences", 
        }
        
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids:
                    input_stream_ids[default_id] = default_stream

        super(VAELatentTraversalQueryModule, self).__init__(
            id=id,
            type="VAELatentTraversalQueryModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logger = input_streams_dict["logger"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        model = input_streams_dict["model"]
        
        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        if epoch % self.config["epoch_period"] == 0 \
        and end_of_epoch:
            image_save_path = logger.path 
            
            VAE = getattr(model,'VAE', None)

            experiences = getattr(model, "experiences", None)
            if experiences is None:
                experiences = input_streams_dict.get("experiences", None)
            if experiences is None:
                raise NotImplementedError 

            if VAE is not None and experiences is not None:
                query_vae_latent_space(VAE, 
                    sample=experiences,
                    path=image_save_path,
                    test=('test' in mode),
                    full=('test' in mode) and self.config.get("traversal", True),
                    idxoffset=epoch,
                    suffix=model.id,
                    use_cuda=True
                )

        return outputs_stream_dict
    
