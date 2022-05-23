from typing import Dict, List, Tuple

import torch

import numpy as np 
import copy 

from .module import Module


def build_LatentToIdAndLabelModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return LatentToIdAndLabelModule(
            id=id,
            config=config, 
            input_stream_ids=input_stream_ids,
    )


class LatentToIdAndLabelModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None,
    ):    
        default_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",

            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            "latent_ohe_representations":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
            "latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
            "indices":"current_dataloader:sample:speaker_indices", 
            
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(LatentToIdAndLabelModule, self).__init__(
            id=id,
            type="LatentToIdAndLabelModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        latent_representations = input_streams_dict["latent_representations"]
        latent_ohe_representations = input_streams_dict["latent_ohe_representations"]
        latent_values_representations = input_streams_dict["latent_values_representations"]
        # (batch_size, nbr_stimuli, temporal_dim, -1 )
        indices = input_streams_dict["indices"]
        
        batch_size = latent_representations.shape[0]
        nbr_latents = latent_representations.shape[-1]
        task_ids = torch.eye(
            nbr_latents, 
            device=latent_representations.device,
        ).unsqueeze(0).repeat(batch_size, 1, 1) #.reshape(-1, nbr_latents)
        # (batch_size x nbr_latents x nbr_latents)
        
        latent_representations = latent_representations.reshape(batch_size, -1)
        for lidx in range(nbr_latents):
            outputs_stream_dict[f'latent_{lidx}_ids'] = task_ids[:, lidx].reshape(batch_size, nbr_latents)
            outputs_stream_dict[f'latent_{lidx}_labels'] = latent_representations[:, lidx].reshape(batch_size, 1)

        return outputs_stream_dict
      

