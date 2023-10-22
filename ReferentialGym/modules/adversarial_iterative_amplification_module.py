from typing import Dict, List 

import itertools
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
import copy 
import wandb 
from tqdm import tqdm, trange

import ReferentialGym
from ReferentialGym.modules import Module
from ReferentialGym.utils.utils import (
    compute_levenshtein_distance,
    compute_levenshtein_distance_for_idx_over_comprange,
)


# Adapted from: 
# https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L37
def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.reshape(-1).tolist())
    
    return t

def compute_levenshtein_distance_for_idx_over_comprange_with_cache(sentences, idx, comprange=None, cache_dict=None):
    if comprange is None: 
      comprange = len(sentences)
    else:
      # ratio:
      comprange = int(comprange*len(sentences))
    levs = []
    s1 = sentences[idx]
    tillidx = min(len(sentences)-1,idx+1+comprange)
    for idx2, s2 in enumerate(sentences[idx+1:tillidx]): 
        if cache_dict is not None \
        and s1 in cache_dict \
        and s2 in cache_dict[s1]:
            lev = cache_dict[s1][s2]
        else:
            lev = compute_levenshtein_distance(s1,s2)
            if s1 not in cache_dict:    cache_dict[s1] = {}
            if s2 not in cache_dict:    cache_dict[s2] = {}
            cache_dict[s1][s2] = cache_dict[s2][s1] = lev
        levs.append(lev)
    return levs 

def compute_levenshtein_distances_parallel(sentences,comprange=None, max_workers=32):
    executor = ProcessPoolExecutor(max_workers=max_workers)
    nbr_sentences = len(sentences)
    indices = list(range(nbr_sentences))
    levs = np.zeros((nbr_sentences, nbr_sentences))
    '''
    for idx1, idx1_levs in tqdm(zip(indices, executor.map(compute_levenshtein_distance_for_idx_over_comprange, itertools.repeat(sentences), indices, itertools.repeat(comprange)))):
        for idx2, ld in enumerate(idx1_levs): 
    '''
    cache_dict = {}
    for idx1 in trange(nbr_sentences):
        idx1_levs = compute_levenshtein_distance_for_idx_over_comprange_with_cache(
            sentences=sentences,
            idx=idx1,
            comprange=comprange,
            cache_dict=cache_dict,
        )
        for idx2, ld in enumerate(idx1_levs):
            levs[idx1, idx2] = ld
            levs[idx2, idx1] = ld
    return levs


class AITAModule(Module):
    def __init__(
        self, 
        id:str,
        config:Dict[str,object]
        ):
        """
        :param id: str defining the ID of the module.
        :param config: Dict of parameters, expecting:
            - "max_workers": Int, number of parallel worker of ProcessPoolExecutor to launch
                when computing levenshtein distances.
            - "update_epoch_period": Int, epoch period when updating the targets.
            - "comprange":  Float, computational range (in percent ratio) of the levenshtein distance
                computation.
            - "init_similarity_ratio": Float, similarity ratio to initialise the 
                distractor sampling scheme with, after one update of the dataset targets.
            - "max_similarity_ratio": Float, similarity ratio maximal threhsold value.
            - "target_unique_prod_ratio": Float, target ratio of unique utterances.
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
            "speaker_exp_indices":"current_dataloader:sample:speaker_indices", 
            #"speaker_exp_indices":"current_dataloader:sample:speaker_exp_indices", 
        }

        super(AITAModule, self).__init__(
            id=id, 
            type="AITAModule",
            config=config,
            input_stream_ids=input_stream_ids
        )
        
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        self.indices = []
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.sentence2class = {}
        self.class_counter = 0
        
        self.updated_dataset_once = False 

        self.max_workers = self.config.get("max_workers", 8)
        self.update_epoch_period = self.config.get("update_epoch_period", 1)
        self.comprange = self.config.get("comprange", 1.0)
        self.similarity_ratio = self.config.get("init_similarity_ratio", 10.0)
        self.max_similarity_ratio = self.config.get("max_similarity_ratio", 100.0)
        self.target_unique_prod_ratio = self.config.get("target_unique_prod_ratio", 100.0)
        
        self.updated_distractor_sampling_likelihoods = None
        self.init_done = False 
        
    def update_similarity_ratio(self, input_streams_dict:Dict[str,object]):
        dataset = input_streams_dict['dataset']
        dataset.kwargs['distractor_sampling'] = f"similarity-{self.similarity_ratio}"

    def update_distractor_sampling_likelihoods(self, input_streams_dict:Dict[str,object]):
        dataset = input_streams_dict['dataset']
        dataset.distractor_sampling_likelihoods = self.updated_distractor_sampling_likelihoods

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
        dataset = input_streams_dict['dataset']
        
        speaker_sentences_widx = input_streams_dict["speaker_sentences_widx"]
        speaker_exp_indices = input_streams_dict["speaker_exp_indices"]

        #if "train" in mode \
        if it_step == 0:
            # Record speaker's sentences:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                speaker_widx = input_streams_dict["speaker_sentences_widx"].cpu().detach()
                batch_size = speaker_widx.shape[0]
                speaker_widx = speaker_widx.reshape(batch_size, -1).numpy()
                indices = input_streams_dict["speaker_exp_indices"]
                indices = indices.cpu().detach().reshape(-1).numpy().tolist()
                for bidx, didx in enumerate(indices):
                    sentence = _hashable_tensor(speaker_widx[bidx])
                    if sentence not in self.sentence2class:
                        self.sentence2class[sentence] = self.class_counter
                        self.class_counter += 1
                    self.speaker_sentences[didx] = sentence
                    self.indices.append(int(didx))

            # Is it the end of the epoch?
            end_of_epoch = all([input_streams_dict[key] for key in self.end_of_])
            
            if end_of_epoch \
            and 'test' in mode:
                set_indices = set(self.indices)
                #all_sentences = np.asarray(list(self.speaker_sentences.values()))
                # Unclear whether the above will order proprely...
                all_sentences = np.asarray([self.speaker_sentences[idx] for idx in range(len(set_indices))])
                _, idx_unique_sentences = np.unique(all_sentences, axis=0, return_index=True)
                #unique_prod_ratio = self.class_counter / len(set_indices) * 100.0
                unique_prod_ratio = len(idx_unique_sentences) / len(self.speaker_sentences) * 100.0
                
                '''
                # update similarity ratio to reach target:
                ratio_diff = self.target_unique_prod_ratio - unique_prod_ratio
                # [-100 ; 100]
                self.similarity_ratio += 0.1*ratio_diff
                self.similarity_ratio = max(0.0, min(self.similarity_ratio, self.max_similarity_ratio))
                '''
                
                logs_dict[f"{mode}/{self.id}/NbrUniqueSentences"] = len(idx_unique_sentences)
                logs_dict[f"{mode}/{self.id}/NonAmbiguousProduction"] = unique_prod_ratio
                '''
                logs_dict[f"{mode}/{self.id}/DistractorsSamplingSimilarityRatio"] = self.similarity_ratio
                
                logs_dict[f"{mode}/{self.id}/DistractorsSamplingSimilarity/Effective"] = 1.0 if self.updated_dataset_once else 0.0
                '''
                
                '''
                if self.updated_dataset_once:
                   self.update_similarity_ratio(input_streams_dict) 
                '''
                
                if self.updated_distractor_sampling_likelihoods is None \
                or np.any(self.updated_distractor_sampling_likelihoods.shape != dataset.distractor_sampling_likelihoods.shape) \
                or np.any(self.updated_distractor_sampling_likelihoods != dataset.distractor_sampling_likelihoods):
                    self.init_done = False

                if self.init_done and (epoch+1) % self.update_epoch_period != 0:
                    return outputs_dict
                
                self.updated_dataset_once = True

                # Compute Levenshtein distances:
                all_sentences_hash = [_hashable_tensor(t) for t in all_sentences]
                lev_dists = compute_levenshtein_distances_parallel(
                    sentences=all_sentences_hash,
                    max_workers=self.max_workers,
                    comprange=self.comprange,
                )
                
                self.updated_distractor_sampling_likelihoods = lev_dists.astype(float)
                self.update_distractor_sampling_likelihoods(input_streams_dict) 
                
                self.init_done = True 

                # WARNING: if there is only one class, then sampling is impossible
                # in descriptive mode when the target should not be retained as the
                # target class is to be excepted from sampling.
                # Thus, we do not update the classes if there is only one language class:
                if False: # len(idx_unique_sentences) != 1:
                    raise NotImplementedError
                    # The following aims to compute classes for each stimuli,
                    # based on their related speaker's description utterance.
                    # This later enables the Dataset to sample (speaker-estimated)
                    # object-centric target (along the distractors...)
                    # stimuli to replace the target stimuli in the listener sample.
                    # update dataset:
                    dataset = input_streams_dict["dataset"]
                    ## assumes DualLabeledDataset...
                    current_target_indices = dataset.train_classes
                    current_mode2offsetidx2class = dataset.mode2offsetidx2class

                    new_train_classes = {}
                    new_mode2offsetidx2class = {'train':{}, 'test':current_mode2offsetidx2class['test']}
                    
                    '''
                    WARNING: due to the dataset effective length,
                    we need to regularise the mode2offsetidx2class element,
                    in order for the DualLabeledDataset to be able to
                    use it despite being agnostic of the original dataset
                    length, and relying solely on the effective length.
                    '''
                    original_dataset_length = len(dataset.datasets['train'].indices)
                    effective_dataset_length = len(dataset.datasets['train'])
                    max_length_factor = 1 + effective_dataset_length // original_dataset_length

                    missing_indices = set(range(original_dataset_length))
                    missing_indices = missing_indices.difference(set_indices)
                    complete_list_indices = list(set_indices)+list(missing_indices)
                    complete_list_effective_indices = list(range(effective_dataset_length))
                    for didx in complete_list_effective_indices:
                        original_didx = didx % original_dataset_length 
                        # Due to ObverterSamplingScheme,
                        # it is likely that not all indices will be seen through out an epoch:
                        if original_didx in set_indices:
                            cl = self.sentence2class[self.speaker_sentences[original_didx]]
                        else:
                            '''
                            WARNING: in order to reuse the previous classes,
                            we need to make sure that those indices do not clash with 
                            the new sentence2class indices, thus we add an offset:
                            '''
                            offset = len(self.sentence2class)
                            cl = offset + current_mode2offsetidx2class['train'][original_didx] # or didx here should be similar...
                        if cl not in new_train_classes: new_train_classes[cl] = []
                        new_train_classes[cl].append(didx)
                        new_mode2offsetidx2class['train'][didx] = cl

                    dataset.train_classes = new_train_classes 

                    '''
                    WARNING: we need to apply an index offset :
                    Should it rely on effective length or original length?
                    The offsetted index is used in mode2offsetidx2class, thus
                    it is used by the DualLabeledDataset, which rely on the 
                    effective length.
                    Thus, we should add as offset the effective dataset length:
                    '''
                    # test_idx_offset = len(dataset.datasets['train'].indices)
                    test_idx_offset = effective_dataset_length

                    new_test_classes = {}
                    for idx in range(len(dataset.datasets['test'])):
                        if hasattr(dataset.datasets['test'], 'getclass'):
                            cl = dataset.datasets['test'].getclass(idx)
                        else :
                            _, cl = dataset.datasets['test'][idx]
                        if cl not in new_test_classes: new_test_classes[cl] = []
                        new_test_classes[cl].append(test_idx_offset+idx)
                        new_mode2offsetidx2class['test'][test_idx_offset+idx] = cl
                
                    # Adding the train classes to the test classes so that we can sample
                    # distractors from the train set:
                    for cl in new_train_classes:
                        if cl not in new_test_classes:
                            new_test_classes[cl] = []
                        for idx in new_train_classes[cl]:
                            new_test_classes[cl].append(idx)
                            new_mode2offsetidx2class['test'][idx] = cl
                
                    dataset.test_classes = new_test_classes
                    dataset.mode2offsetidx2class = new_mode2offsetidx2class
                ### END IF

                # Reset:
                self.reset_storage()
            
            elif self.updated_distractor_sampling_likelihoods is not None \
            and (np.any(self.updated_distractor_sampling_likelihoods.shape != dataset.distractor_sampling_likelihoods.shape) \
            or np.any(self.updated_distractor_sampling_likelihoods != dataset.distractor_sampling_likelihoods)):
                #Updating the likelihoods in the training dataset:
                self.update_distractor_sampling_likelihoods(input_streams_dict) 
        
        return outputs_dict

    def reset_storage(self):
        self.speaker_sentences = {} #from dataset's idx to sentence.
        self.indices = []
        self.sentence2class = {}
        self.class_counter = 0
        return 

