from typing import Dict, List 

import torch
import numpy as np
import copy 

import ReferentialGym
from ReferentialGym.modules import Module
from ReferentialGym.utils.utils import (
    compute_levenshtein_distance,
)


def compute_levenshtein_distances_with_cache(
    idx2sentences1,
    idx2sentences2,
):
    cache_dict = {} 
    levs = []
    for idx1, s1 in idx2sentences1.items(): 
        if idx1 not in idx2sentences2:  continue
        s2 = idx2sentences2[idx1]
        if s1 in cache_dict \
        and s2 in cache_dict[s1]:
            lev = cache_dict[s1][s2]
        else:
            lev = compute_levenshtein_distance(s1,s2)
            if s1 not in cache_dict:    cache_dict[s1] = {}
            if s2 not in cache_dict:    cache_dict[s2] = {}
            cache_dict[s1][s2] = cache_dict[s2][s1] = lev
        levs.append(lev)
    return levs 


class JaccardSimilarityMetricModule(Module):
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

        super(JaccardSimilarityMetricModule, self).__init__(
            id=id, 
            type="JaccardSimilarityMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        self.indices = []
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.speaker_ids = []
        self.speaker2idx2widx = {}
        self.new_data = {}
        
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
        speaker_id = input_streams_dict["speaker_id"]
        if speaker_id not in self.new_data:   self.new_data[speaker_id] = 0
        dataset = input_streams_dict["dataset"]
        dataset_size = dataset.size()
        
        if "train" in mode \
        and it_step == 0:
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
                    self.indices.append(int(didx))
                    if didx not in self.speaker2idx2widx[speaker_id]: self.new_data[speaker_id] += 1
                    self.speaker2idx2widx[speaker_id][didx] = sentence
            
            # Is it the end of the epoch?
            end_of_epoch = all([input_streams_dict[key] for key in self.end_of_])
            
            if end_of_epoch:
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
                        
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}_{speaker_id2}/JaccardSimilarity"] = jaccard_sim
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}_{speaker_id2}/MeanLevDist"] = np.mean(levs)
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}_{speaker_id2}/OverlapRatio"] = ratio
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}_{speaker_id2}/DatasetRatio"] = dataset_ratio
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}/NbrNewData"] = self.new_data[speaker_id1]
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id2}/NbrNewData"] = self.new_data[speaker_id2]
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id1}/NewDataRatio"] = self.new_data[speaker_id1]/dataset_size
                        logs_dict[f"{mode}/{self.id}/JaccardSimilarityMetric/{speaker_id2}/NewDataRatio"] = self.new_data[speaker_id2]/dataset_size
                # Reset:
                self.reset_storage()
                
        return outputs_dict

    def reset_storage(self):
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.indices = []
        self.speaker_ids = []
        self.new_data = {}
        return 

