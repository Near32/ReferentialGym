from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module

def build_TopographicSimilarityMetricModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return TopographicSimilarityMetricModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class TopographicSimilarityMetricModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        input_stream_ids = {
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
            
            "current_speaker":"modules:current_speaker:ref:ref_agent",
            "sentences_widx":"modules:current_speaker:sentences_widx",
            "current_listener":"modules:current_listener:ref:ref_agent",
        }

        super(TopographicSimilarityMetricModule, self).__init__(id=id,
                                                 type="TopographicSimilarityMetric",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.whole_epoch_sentences = []
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        epoch = input_streams_dict["epoch"]        
        mode = input_streams_dict["mode"]
        
        if epoch % self.config["epoch_period"] == 0:
            speaker = input_streams_dict["current_speaker"]
            listener = input_streams_dict["current_listener"]
            
            # Store current speaker's sentences:
            sentences = []
            speaker_sentences_widx = input_streams_dict["sentences_widx"]
            batch_size = speaker_sentences_widx.shape[0]
            for sidx in range(batch_size):
                sentences.append("".join([chr(97+int(s.item())) for s in speaker_sentences_widx[sidx] ]))    
            for sentence in sentences:  
                self.whole_epoch_sentences.append(sentence.replace(chr(97+self.config["vocab_size"]), ""))

            # Is it the end of the epoch?
            end_of_epoch = all([
              input_streams_dict[key]
              for key in self.end_of_]
            )
            
            # If so, let us average over every value and save it:
            if end_of_epoch:
                logger = input_streams_dict["logger"]

                max_nbr_samples = None
                if self.config["fast"]:  
                    max_nbr_samples = int(len(self.whole_epoch_sentences)*0.1)

                topo_sims, pvalues, unique_prod_ratios = logger.measure_topographic_similarity(sentences_key="sentences_widx",
                                                                           features_key="exp_latents",
                                                                           max_nbr_samples=max_nbr_samples,
                                                                           verbose=self.config["verbose"],
                                                                           max_workers=self.config["parallel_TS_computation_max_workers"])
                topo_sims_v, pvalues_v, unique_prod_ratios_v = logger.measure_topographic_similarity(sentences_key="sentences_widx",
                                                                           features_key="exp_latents_values",
                                                                           max_nbr_samples=max_nbr_samples,
                                                                           verbose=self.config["verbose"],
                                                                           max_workers=self.config["parallel_TS_computation_max_workers"])
                feat_topo_sims, feat_pvalues, _ = logger.measure_topographic_similarity(sentences_key="sentences_widx",
                                                                           features_key="temporal_features",
                                                                           max_nbr_samples=max_nbr_samples,
                                                                           verbose=self.config["verbose"],
                                                                           max_workers=self.config["parallel_TS_computation_max_workers"])
                
                for agent_id in topo_sims:
                    logs_dict[f"{mode}/{self.id}/TopographicSimilarity/{agent_id}"] = topo_sims[agent_id]*100.0
                    logs_dict[f"{mode}/{self.id}/TopographicSimilarity-NonAmbiguousProduction/{agent_id}"] = unique_prod_ratios[agent_id]
                    logs_dict[f"{mode}/{self.id}/TopographicSimilarity-PValues/{agent_id}"] = pvalues[agent_id]
                for agent_id in topo_sims_v:
                    logs_dict[f"{mode}/{self.id}/TopographicSimilarity_withValues/{agent_id}"] =  topo_sims_v[agent_id]*100.0
                    logs_dict[f"{mode}/{self.id}/TopographicSimilarity_withValues-PValues/{agent_id}"] = pvalues_v[agent_id]
                for agent_id in feat_topo_sims:
                    logs_dict[f"{mode}/{self.id}/FeaturesTopographicSimilarity/{agent_id}"] = feat_topo_sims[agent_id]*100.0

                # Reset epoch storages:
                self.whole_epoch_sentences = []

        return outputs_stream_dict
    