from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim 

import copy
import numpy as np 
import math
import sklearn 
from functools import partial 
import cv2 as cv
import matplotlib.pyplot as plt

import wandb 

from .module import Module
from ReferentialGym.utils import compute_cosine_sim
eps = 1e-20

"""
"""
# Adapted from: 
# https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L37
def ht(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t
    if isinstance(t, list):
        return tuple(t)

    try:
        t = t.item()
    except:
        t = tuple(t.reshape(-1).tolist())
    
    return t

    
def build_CompactnessAmbiguityMetricModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return CompactnessAmbiguityMetricModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class CompactnessAmbiguityMetricModule(Module):
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

            "end_of_dataset":"signals:end_of_dataset",  
            # boolean: whether the current batch/datasample is the last of the current dataset/mode.
            "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
            # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
            # is the last of the current sequence of repetition.
            "end_of_communication":"signals:end_of_communication",
            # boolean: whether the current communication round is the last of 
            # the current dialog.
            "dataset":"current_dataset:ref",

            "model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            #"top_view":"current_dataloader:sample:speaker_top_view", 
            #"agent_pos_in_top_view":"current_dataloader:sample:speaker_agent_pos_in_top_view", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            "indices":"current_dataloader:sample:speaker_indices", 
            
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(CompactnessAmbiguityMetricModule, self).__init__(
            id=id,
            type="CompactnessAmbiguityMetricModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.sanity_check_shuffling = self.config['sanity_check_shuffling']

        # Default = 0.0
        self.repr_dim_filtering_threshold = self.config["threshold"]
        
        self.idx2w = self.config['idx2w']

        current_random_state = np.random.get_state()
        np.random.seed(self.config['random_state_seed'])
        self.random_state = np.random.get_state()
        np.random.set_state(current_random_state)

        self.experiences = {}
        self.make_visualisation = False
        if "top_view" in self.input_stream_ids:
            self.make_visualisation = True
            self.top_views = {}
            self.agent_pos_in_top_views = {}
            
        self.representations = {}
        self.latent_representations = {}
        self.indices = []

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if not(epoch % self.config["epoch_period"] == 0):
            return outputs_stream_dict

        if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
            experiences = input_streams_dict["experiences"].cpu().detach().squeeze().numpy()
            if self.make_visualisation:
                top_views = input_streams_dict["top_view"].cpu().detach().squeeze().numpy()
                agent_pos_in_top_views = input_streams_dict["agent_pos_in_top_view"].cpu().detach().squeeze().numpy()

            representations = input_streams_dict["representations"].cpu().detach().squeeze().numpy()
            latent_representations = input_streams_dict["latent_representations"].cpu().detach().squeeze().numpy()
            indices = input_streams_dict["indices"].cpu().detach().squeeze().numpy()
            
            for idx, tidx in enumerate(indices.tolist()):
                self.experiences[tidx] = experiences[idx]
                if self.make_visualisation:
                    self.top_views[tidx] = top_views[idx]
                    self.agent_pos_in_top_views[tidx] = agent_pos_in_top_views[idx]
                self.representations[tidx] = representations[idx]
                self.latent_representations[tidx] = latent_representations[idx]
            self.indices.append(indices)

        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        if not(end_of_epoch and 'test' in mode): 
            return outputs_stream_dict

        self.experiences = np.stack(list(self.experiences.values()), axis=0)
        if self.make_visualisation:
            self.top_views = np.stack(list(self.top_views.values()), axis=0)
            self.agent_pos_in_top_views = np.stack(list(self.agent_pos_in_top_views.values()), axis=0)
        self.representations = np.stack(list(self.representations.values()), axis=0)
        
        latent_shape = self.latent_representations[0].shape
        self.latent_representations = np.stack(list(self.latent_representations.values()), axis=0)
        self.indices = np.concatenate(self.indices, axis=0).reshape(-1)

        # Make sure every index is only seen once:
        self.original_indices = self.indices
        sorted_unique_indices, sampling_indices = np.unique(self.indices, return_index=True)
        
        assert len(sorted_unique_indices) == len(self.original_indices)

        self.experiences = self.experiences[sampling_indices]
        if self.make_visualisation:
            self.top_views = self.top_views[sampling_indices]
            self.agent_pos_in_top_views = self.agent_pos_in_top_views[sampling_indices]
        self.representations = self.representations[sampling_indices]
        self.latent_representations = self.latent_representations[sampling_indices]
        
        if self.sanity_check_shuffling:
            rng = np.random.default_rng()
            perm = rng.permutation(len(self.experiences))
            #self.experiences = self.experiences[perm]
            self.representations = self.representations[perm]
            #self.latent_representations = self.latent_representations[perm]

        # From here on, the previous tensors are ordered with respect to
        # their actual position in the dataset :
        # i.e. self.experiences[i] corresponds to the i-th element of the dataset.
        # Thus, assumming the i-th element of the dataset is the i-th experience tuple
        # sampled, we can be sure that the clustering is done over episode timesteps.

        all_sentences = [ht(s) for s in self.representations.tolist()]
        sentence_length = len(all_sentences[0]) #.shape[0]
        unique_sentences = set(all_sentences) #np.unique(all_sentences, axis=0)
        
        if self.make_visualisation:
            cluster_change_indices = []
            episode_change_indices = []
            cluster_sentences = []
            prev_sentence = None

        per_unique_sentence_stats = {}
        previous_sentence = None
        compactness_count = 0
        for idx, sentence in enumerate(all_sentences):
            if self.make_visualisation:
                if prev_sentence != sentence:
                    cluster_sentences.append(sentence)
                    cluster_change_indices.append(idx)
                prev_sentence = sentence
                terminal = (self.latent_representations[idx][2]==0)
                if terminal:
                    # end-of-episode indices transition are recorded.
                    episode_change_indices.append(idx)

            if sentence not in per_unique_sentence_stats:
                per_unique_sentence_stats[sentence] = {
                    'occ_indices': [],
                    'compactness_counts': [],
                }
            per_unique_sentence_stats[sentence]['occ_indices'].append(idx)
            
            if previous_sentence == sentence:
                compactness_count += 1
            else:
                if previous_sentence is not None:
                    per_unique_sentence_stats[previous_sentence]['compactness_counts'].append(compactness_count)

                compactness_count = 1
            previous_sentence = sentence
        # Regularise the last sentence:
        per_unique_sentence_stats[sentence]['compactness_counts'].append(compactness_count)
        
        if self.make_visualisation:
            visualisation = self.draw(
                top_views=self.top_views,
                agent_pos_in_top_views=self.agent_pos_in_top_views,
                cluster_change_indices=cluster_change_indices,
                episode_change_indices=episode_change_indices,
                cluster_sentences=cluster_sentences,
            )

        mode = input_streams_dict["mode"]
        dataset = input_streams_dict["dataset"].datasets[mode]
        logger = input_streams_dict["logger"]

        
        nbr_frames = self.config["kwargs"]['task_config']['nbr_frame_stacking'] #succ_s[bidx].shape[0]//4
        frame_depth = self.config["kwargs"]['task_config']['frame_depth']
        
        columns = [f"idx"]
        columns += [f"token{idx}" for idx in range(sentence_length)]
        
        if self.config['show_stimuli']:
            columns += ["stimulus"]
        
        columns += ["nbr_compact_segment"]
        columns += ["min_compactness", "max_compactness"]
        columns += [f"latent{idx}" for idx in range(latent_shape[0])]
        
        #self.sample_table = wandb.Table(columns=columns) 
                    
        min_sum = 0
        max_sum = 0
        normalizer = len(all_sentences)
        for idx, sentence in enumerate(all_sentences):
            data = []

            data.append(idx)

            for widx in sentence:
                word = self.idx2w[widx]
                data.append(word)

            '''
            if self.config['show_stimuli']:
                exp = self.experiences[idx]
                stimulus_t = exp.reshape(nbr_frames,frame_depth, *exp.shape[-2:])
                stimulus_t = stimulus_t[:,:3]*255
                stimulus_t = stimulus_t.astype(np.uint8)
                stimulus_t = wandb.Video(stimulus_t, fps=1, format="gif")
                data.append(stimulus_t)
            '''

            stats = per_unique_sentence_stats[sentence]

            nbr_compact_segment = len(stats['compactness_counts'])
            data.append(nbr_compact_segment)

            min_compactness = min(stats['compactness_counts'])
            min_sum += min_compactness
            max_compactness = max(stats['compactness_counts'])
            max_sum += max_compactness
            data.append(min_compactness)
            data.append(max_compactness)

            for lidx in self.latent_representations[idx]:
                data.append(lidx.item())

            #self.sample_table.add_data(*data)
                    
        #wandb.log({f"{mode}/{self.id}/PerEpoch/CompactnessTable":self.sample_table}, commit=False)

        ## Compute Compactness Score:
        list_compactness_counts = []
        for us, stat in per_unique_sentence_stats.items():
            for cc in stat['compactness_counts']:
                list_compactness_counts.append(cc)
        values = np.asarray(list_compactness_counts)

        mean_compactness_counts = values.mean()
        std_compactness_counts = values.std()
        median_value = np.nanpercentile(
            values,
            q=50,
            axis=None,
            method="nearest"
        )
        q1_value = np.nanpercentile(
            values,
            q=25,
            axis=None,
            method="lower"
        )
        q3_value = np.nanpercentile(
            values,
            q=75,
            axis=None,
            method="higher"
        )
        iqr = q3_value-q1_value
        
        '''
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Mean"] = mean_compactness_counts
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Std"] = std_compactness_counts
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Min"] = min(values)
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Max"] = max(values)
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Median"] = median_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Q1"] = q1_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Q3"] = q3_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/IQR"] = iqr
        '''

        mean_min_compactness = float(min_sum) / normalizer
        #logs_dict[f"{mode}/{self.id}/CompactnessCounts/Minimal/Mean"] = mean_min_compactness
        mean_max_compactness =  float(max_sum) / normalizer
        #logs_dict[f"{mode}/{self.id}/CompactnessCounts/Maximal/Mean"] = mean_max_compactness

        list_nbr_compact_segment = [len(ps['compactness_counts']) for ps in per_unique_sentence_stats.values()]
        mean_nbr_compact_segment = sum(list_nbr_compact_segment)/len(list_nbr_compact_segment)
        min_nbr_compact_segment = min(list_nbr_compact_segment)
        max_nbr_compact_segment = max(list_nbr_compact_segment)
        values = np.asarray(list_nbr_compact_segment)
        std_nbr_compact_segment = values.std()
        
        median_value = np.nanpercentile(
            values,
            q=50,
            axis=None,
            method="nearest"
        )
        q1_value = np.nanpercentile(
            values,
            q=25,
            axis=None,
            method="lower"
        )
        q3_value = np.nanpercentile(
            values,
            q=75,
            axis=None,
            method="higher"
        )
        iqr = q3_value-q1_value
        
        '''
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Mean"] = mean_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Std"] = std_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Min"] = min_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Max"] = max_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Median"] = median_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Q1"] = q1_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Q3"] = q3_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/IQR"] = iqr
        '''

        average_max_compactness_count = len(self.representations) / len(unique_sentences)
        logs_dict[f"{mode}/{self.id}/CompactnessAmbiguity/NbrRepresentations"] = len(self.representations) 
        logs_dict[f"{mode}/{self.id}/CompactnessAmbiguity/NbrUniqueSentences"] = len(unique_sentences) 
        logs_dict[f"{mode}/{self.id}/CompactnessAmbiguity/AverageMaxCompactnessCount"] = average_max_compactness_count 

        percentages = [0.0306125, 0.06125, 0.125, 0.25, 0.5, 0.75]
        thresholds = [1+max(1, math.ceil(percent*average_max_compactness_count))
            for percent in percentages]
        for tidx, threshold in enumerate(thresholds):
            logs_dict[f"{mode}/{self.id}/CompactnessAmbiguity/Threshold{tidx}"] = threshold 
            nbr_max_compactness_count_greater_than_threshold = len([
                count for count in list_compactness_counts if count >= threshold]
            )
            compactness_score = float(nbr_max_compactness_count_greater_than_threshold) / len(list_compactness_counts)*100.0

            logs_dict[f"{mode}/{self.id}/CompactnessAmbiguity/Score@Threshold{tidx}"] = compactness_score 
        logs_dict[f"{mode}/{self.id}/Ambiguity"] = (1.0-(len(unique_sentences)/len(all_sentences)))*100.0 

        self.experiences = {}
        if self.make_visualisation:
            self.top_views = {}
            self.agent_pos_in_top_views = {}
        self.representations = {}
        self.latent_representations = {}
        self.indices = []
        
        return outputs_stream_dict

    def draw(
        self,
        top_views,
        agent_pos_in_top_views,
        cluster_change_indices,
        episode_change_indices,
        cluster_sentences,
        color_start=np.array([0,0,1,1]),
        color_end=np.array([1,0,0,1]),
    ):
        log_dict = {} 

        cluster_sentence_idx = 0
        episode_idx = 0
        start_episode_idx = 0
        for end_episode_idx in episode_change_indices:
            visualisation = top_views[(start_episode_idx+end_episode_idx)//2]
            visualisation = cv.cvtColor(visualisation, cv.COLOR_RGB2RGBA)
            path_vis = visualisation.copy()
            perspective_vis = visualisation.copy()
            
            drawn_sentences = []
            nbr_cluster_changes = 0
            episode_length = end_episode_idx-start_episode_idx+2
            for idx in range(start_episode_idx, end_episode_idx+1):
                offset = idx-start_episode_idx
                color_blend = float(offset) / episode_length
                color = 255*((1-color_blend)*color_start+color_blend*color_end)
                radius = 3

                agent_pos = agent_pos_in_top_views[idx]
                if idx in cluster_change_indices \
                or idx==start_episode_idx:
                    current_sentence = cluster_sentences[cluster_sentence_idx]
                    if idx in cluster_change_indices:
                        cluster_sentence_idx += 1
                    nbr_cluster_changes += 1

                    # If that sentence is already drawn,
                    # then we skip it.
                    # Otherwise it is going to be intractable:
                    if current_sentence not in drawn_sentences:
                        drawn_sentences.append(current_sentence)
                        
                        # Draw triangle:
                        persp_color = color.copy()
                        persp_color[-1] = 192
                        prespective_vis = cv.drawContours(
                            perspective_vis,
                            contours=[agent_pos.astype(int)],
                            contourIdx=-1, # draw all contours
                            color=persp_color.astype(int).tolist(),
                            thickness=-1,
                        )
                        
                        # Draw triangle's contour:
                        persp_color = color.copy()
                        persp_color[-1] = 223
                        prespective_vis = cv.drawContours(
                            perspective_vis,
                            contours=[agent_pos.astype(int)],
                            contourIdx=-1, # draw all contours
                            color=persp_color.astype(int).tolist(),
                            thickness=2,
                        )

                        # This point is drawn differently:
                        #color = color_start
                        radius = 5
                
                # Draw points
                path_vis = cv.circle(
                    path_vis,
                    center=agent_pos[0].astype(int), 
                    radius=radius, 
                    color=color.tolist(), 
                    thickness=-1, # Filled
                )
            
            start_episode_idx = end_episode_idx+2
            episode_idx += 1
            
            alpha=0.25
            visualisation = cv.addWeighted(
                visualisation, 
                alpha, 
                path_vis, 
                1-alpha, 
                0,
            )
                     
            alpha=0.75
            visualisation = cv.addWeighted(
                visualisation, 
                alpha, 
                perspective_vis, 
                1-alpha, 
                0,
            )
            
            #plt.imshow(perspective_vis)
            #plt.savefig(f'./persp_vis_episode{episode_idx}.png')
            log_dict[f"{self.id}/PerspVis-Episode{episode_idx}"] = wandb.Image(
                perspective_vis,
                caption=f"",
            )
            #plt.imshow(path_vis)
            #plt.savefig(f'./path_vis_episode{episode_idx}.png')
            log_dict[f"{self.id}/PathVis-Episode{episode_idx}"] = wandb.Image(
                path_vis,
                caption=f"",
            )

            #plt.imshow(visualisation)
            #plt.savefig(f'./vis_episode{episode_idx}.png')
            log_dict[f"{self.id}/Vis-Episode{episode_idx}"] = wandb.Image(
                visualisation,
                caption=f"EpisodeLength={episode_length}-NbrUniqueCompactClusters={len(drawn_sentences)}/NbrClusterChanges={nbr_cluster_changes}",
            )
        
        wandb.log(log_dict, commit=False)
        
        return visualisation
 
