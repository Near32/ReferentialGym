from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
from tqdm import tqdm

from .module import Module


def build_PerLatentAccuracyMetricModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return PerLatentAccuracyMetricModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids,
    )


class PerLatentAccuracyMetricModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None,
    ):
        raise NotImplementedError
        # TODO: debug see below...

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
            
            "accuracy":"modules:current_listener:accuracy",
            "test_latents_mask":"current_dataloader:sample:speaker_exp_test_latents_masks",
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_stream, default_id in default_input_stream_ids.items():
                if default_id not in input_stream_ids.values():
                    input_stream_ids[default_stream] = default_id

        super(PerLatentAccuracyMetricModule, self).__init__(
            id=id,
            type="PerLatentAccuracyMetricModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.accuracies = []
        self.test_latents_masks = []
        self.dataset_id = self.config['dataset_id']

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if epoch==0 or mode not in self.config["modes"]:
            return outputs_stream_dict

        accuracy = input_streams_dict["accuracy"]
        self.accuracies.append(accuracy.cpu().detach().numpy())
        # (nbr_element++, batch_size (may vary..),) 
        test_latents_mask = input_streams_dict["test_latents_mask"]
        self.test_latents_masks.append(test_latents_mask.cpu().detach().squeeze().numpy())
        # (nbr_element++, batch_size (may vary..), nbr_latents) 
        
        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        if end_of_epoch:
            nbr_elements =  len(self.accuracies)
            nbr_latents = self.test_latents_masks[0].shape[-1]
            latents_indices = np.arange(nbr_latents)

            logger = input_streams_dict["logger"]

            co_occ_count_matrix =  np.zeros((nbr_latents, nbr_latents))
            co_occ_result_matrix = np.zeros((nbr_latents, nbr_latents))
            nbr_tests = 0
            for batch_accuracies, batch_test_latents_mask in tqdm(zip(self.accuracies, self.test_latents_masks)):
                batch_size = batch_accuracies.shape[0]
                nbr_tests += batch_size
                for batch_idx in range(batch_size):
                    #TODO: debug here index out of bounds...
                    tested_latents_indices = latents_indices[batch_test_latents_mask[batch_idx]>0]
                    for idxl1 in tested_latents_indices:
                        for idxl2 in tested_latents_indices:
                            if idxl2 >= idxl1:
                                co_occ_count_matrix[idxl2, idxl1] += 1
                                co_occ_result_matrix[idxl2, idxl1] += batch_accuracies[batch_idx]
                                # only fill up the lower triangle...

            marg_p_latents = {}
            joint_p_latents = np.zeros_like(co_occ_result_matrix)
            safe_divider = (co_occ_count_matrix>0)*co_occ_count_matrix+(co_occ_count_matrix==0)*np.ones_like(co_occ_count_matrix)
            for idx_latent in range(nbr_latents):
                
                joint_p_latents[:,idx_latent] = co_occ_result_matrix[:,idx_latent]/safe_divider[:,idx_latent]
                marg_p_latents[idx_latent] = joint_p_latents[idx_latent,idx_latent]
                
                logs_dict[f"{mode}/{self.id}/{self.dataset_id}/TestLatentValues/MarginalAccuracy/Latent{idx_latent}"] = marg_p_latents[idx_latent]
                
                nbr_test4latent = co_occ_count_matrix[idx_latent+1:, idx_latent].sum()
                if nbr_test4latent == 0:    continue
                
                for idx_latent2 in range(nbr_latents):
                    if idx_latent2 > idx_latent:
                        logs_dict[f"{mode}/{self.id}/{self.dataset_id}/TestLatentValues/JointAccuracy/Latent-{idx_latent}-{idx_latent2}"] = joint_p_latents[idx_latent2, idx_latent]
            
            self.accuracies = []  
            self.test_latents_mask = []
            
        return outputs_stream_dict
    
