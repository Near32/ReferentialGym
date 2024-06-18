from typing import Dict, List 

import torch
import numpy as np
import copy 

import ReferentialGym
from ReferentialGym.modules import Module
from ReferentialGym.utils.utils import (
    compute_levenshtein_distance,
)
from ReferentialGym.modules.jaccard_similarity_metric_module import compute_levenshtein_distances_with_cache


class LanguageDynamicMetricModule(Module):
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

            "speaker_sentences_widx":"modules:current_speaker:sentences_widx", 
            "speaker_indices":"current_dataloader:sample:speaker_indices", 
            "speaker_exp_indices":"current_dataloader:sample:speaker_exp_indices", 
            "speaker_id":"modules:current_speaker:ref:ref_agent:id"
        }

        super(LanguageDynamicMetricModule, self).__init__(
            id=id, 
            type="LanguageDynamicMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.speaker_ids = []
        self.speaker2idx2widx = {}
        
    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L37
    def _hashable_tensor(self, t):
        if isinstance(t, tuple):
            return t
        if isinstance(t, int):
            return t

        try:
            t = t.item()
        except:
            t = tuple(t.reshape(-1).tolist())
        
        return t
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        
        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
        """

        outputs_dict = {}

        logs_dict = input_streams_dict["logs_dict"]
        epoch = input_streams_dict["epoch"]
        mode = input_streams_dict["mode"]
        it_step = input_streams_dict["it_step"]
        
        speaker_sentences_widx = input_streams_dict["speaker_sentences_widx"]
        speaker_exp_indices = input_streams_dict["speaker_exp_indices"]
        speaker_id = 'new_speaker' #input_streams_dict["speaker_id"]
        dataset = input_streams_dict["dataset"]
        dataset_size = dataset.size()

        if it_step != 0:
            return outputs_dict

        # Record speaker's sentences and id:
        if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
            self.speaker_ids.append(speaker_id)
            speaker_widx = input_streams_dict["speaker_sentences_widx"].cpu().detach()
            batch_size = speaker_widx.shape[0]
            speaker_widx = speaker_widx.reshape(batch_size, -1).numpy()
            indices = input_streams_dict["speaker_exp_indices"]
            indices = indices.cpu().detach().reshape(-1).numpy().tolist()
            for bidx, didx in enumerate(indices):
                if speaker_id not in self.speaker2idx2widx: self.speaker2idx2widx[speaker_id] = {}
                sentence = self._hashable_tensor(speaker_widx[bidx])
                self.speaker_sentences[didx] = sentence
                self.speaker2idx2widx[speaker_id][didx] = sentence
        
        # Is it the end of the epoch?
        end_of_epoch = all([input_streams_dict[key] for key in self.end_of_])
        if not(end_of_epoch and 'test' in mode): 
            return outputs_dict

        cache_dict = {} 
        self.speaker_id_set = sorted(list(set(self.speaker_ids)), key=str.lower)
        speaker2levs = {}
        speaker2ratios = {}
        for sp1idx, speaker_id1 in enumerate(self.speaker_id_set):
            speaker2levs[speaker_id1] ={}
            speaker2ratios[speaker_id1] ={}
            for speaker_id2 in self.speaker_id_set[sp1idx:]:
                if speaker_id1 == speaker_id2:  continue
                levs = speaker2levs[speaker_id1][speaker_id2] = compute_levenshtein_distances_with_cache(
                    idx2sentences1=self.speaker2idx2widx[speaker_id1],
                    idx2sentences2=self.speaker2idx2widx[speaker_id2],
                    cache_dict=cache_dict, 
                )
                #speaker2levs[speaker_id2][speaker_id1] = levs
                nbr_levs = len(levs)
                nbr_indices1 = len(self.speaker2idx2widx[speaker_id1])
                nbr_indices2 = len(self.speaker2idx2widx[speaker_id2])
                ratio = 2*nbr_levs/(nbr_indices1+nbr_indices2)
                speaker2ratios[speaker_id1][speaker_id2] = ratio
                #speaker2ratios[speaker_id2][speaker_id1] = ratio
                dataset_ratio = len(levs)/dataset_size
                
                jaccard_sim = [1 if lev==0 else 0 for lev in levs]
                jaccard_sim = np.mean(jaccard_sim)
                
                logs_dict[f"{mode}/{self.id}/LanguageDynamicMetric/{speaker_id1}_{speaker_id2}/JaccardSimilarity"] = jaccard_sim
                logs_dict[f"{mode}/{self.id}/LanguageDynamicMetric/{speaker_id1}_{speaker_id2}/MeanLevDist"] = np.mean(levs)
                logs_dict[f"{mode}/{self.id}/LanguageDynamicMetric/{speaker_id1}_{speaker_id2}/OverlapRatio"] = ratio
                logs_dict[f"{mode}/{self.id}/LanguageDynamicMetric/{speaker_id1}_{speaker_id2}/DatasetRatio"] = dataset_ratio
        # Reset:
        self.update_storage()
        
        return outputs_dict

    def update_storage(self):
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.speaker2idx2widx = {'old_speaker': self.speaker2idx2widx['new_speaker']}
        self.speaker_ids = ['old_speaker']
        return 

