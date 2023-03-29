from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim 

import copy
import numpy as np 
import sklearn 
from functools import partial 

import wandb 

from .module import Module
from ReferentialGym.utils import compute_cosine_sim
eps = 1e-20

"""
"""

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
        
        # Default = 0.0
        self.repr_dim_filtering_threshold = self.config["threshold"]
        
        current_random_state = np.random.get_state()
        np.random.seed(self.config['random_state_seed'])
        self.random_state = np.random.get_state()
        np.random.set_state(current_random_state)

        self.experiences = {}
        self.representations = {}
        self.latent_representations = {}
        self.indices = []

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
    
    def _generate_training_batch(
        self,
        dataset,
        model,
        batch_size,
        nbr_points,
        active_dims,
    ):
        """
        Sample a set of training samples based on a batch of ground-truth data.
        
        Args:
            dataset: dataset to be sampled from.
            model: model that takes observations as input and
                    outputs a dim_representation sized representation for each observation.
            batch_size: Number of points to be used to compute the training_sample.
            nbr_points: Number of points to be sampled for training/evaluation set.
            active_dims: Indexes of active dimensions.
        Returns:
            (num_factors, dim_representation)-sized numpy array with votes.
        
        """
        current_random_state = np.random.get_state()
        np.random.set_state(copy.deepcopy(self.random_state))
        
        representations = []
        latent_representations = []
        i = 0
        while i < nbr_points:
            num_points = min(nbr_points-i, batch_size)
            rep, lrep = self._generate_training_sample(
                dataset=dataset,
                model=model,
                batch_size=num_points, 
                active_dims=active_dims)
            # (batch_size, dim)
            representations.append(rep)
            latent_representations.append(lrep)
            i+= num_points
        
        representations = np.concatenate(representations, axis=0)
        latent_representations = np.concatenate(latent_representations, axis=0)

        self.random_state = copy.deepcopy(np.random.get_state())
        np.random.set_state(current_random_state)
        
        return np.transpose(representations), np.transpose(latent_representations)

    def _generate_training_sample(
        self, 
        dataset, 
        model,
        batch_size, 
        active_dims,
    ):
        """
        Sample a single training sample based on a mini-batch of ground-truth data.
        
        Args:
        dataset: dataset to be sampled from.
        model: model that takes observation as input and
                outputs a representation.
        batch_size: Number of points to generate a sample with.
        active_dims: Indexes of active dimensions.
        
        Returns:
            representation: np.array of size (repr_dim x batch_size)
        
        """
        
        # Select random coordinate to keep fixed.
        if self.config["active_factors_only"]:
            factor_index = np.random.choice(self.active_latent_dims)
        else:
            factor_index = np.random.choice(list(range(self.nbr_factors))) #self.random_state.randint(self.nbr_factors)
        
        if self.config["resample"]:
            # Sample two mini batches of latent variables.
            factors = dataset.sample_factors(
                batch_size, 
                random_state=None, #self.random_state
            )
            # Fix the selected factor across mini-batch.
            factors[:, factor_index] = factors[0, factor_index]
            # Obtain the observations.
            observations = dataset.sample_observations_from_factors(
              factors, 
              random_state=None, #self.random_state
            )
            
            if "preprocess_fn" in self.config:
                observations = self.config["preprocess_fn"](observations)
            
            if hasattr(model,"encodeZ"):
                relevant_representations = model.encodeZ(observations)
            else:
                relevant_representations = model(observations)

            if "postprocess_fn" in self.config:
                relevant_representations = self.config["postprocess_fn"](relevant_representations)
            
            relevant_latent_representations = factors
        else:
            # Sample from the current epoch"s samples the factor value to fix:
            sample_to_fix_factor_value_idx = np.random.choice(np.arange(self.latent_representations.shape[0]))
            factor_value = self.latent_representations[sample_to_fix_factor_value_idx,...,factor_index]
            
            # Sample from the current epoch the indices of relevant samples:
            relevant_samples_indices = [
                it for it, lr in enumerate(self.latent_representations) 
                if lr[...,factor_index]== factor_value
            ]
            
            if len(relevant_samples_indices) < batch_size:
                if self.config["verbose"]:
                    print(f"WARNING: generate_training_sample ::\
                     too few relevant samples: {len(relevant_samples_indices)} < batch_size={batch_size}.\n\
                     Falling back on this value...")
                batch_size = len(relevant_samples_indices)
            # No issue of batch_size = 0 ....
            relevant_samples_indices_sampled = np.random.choice(relevant_samples_indices, 
                size=batch_size,
                replace=False)
            relevant_representations = self.representations[relevant_samples_indices_sampled]
            # (batch_size, repr_dim)
            relevant_latent_representations = self.latent_representations[relevant_samples_indices_sampled]
            # (batch_size, latent_repr_dim)

        return relevant_representations, relevant_latent_representations

    def _compute_entropy(self, rep):
        rep_size = rep.shape[0]
        ent = np.zeros([rep_size,])
        for i in range(rep_size):
            ent[i] = sklearn.metrics.mutual_info_score(rep[i, :], rep[i, :])
        return ent

    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L15
    def entropy_dict(self, freq_table):
        H = 0
        n = sum(v for v in freq_table.values())

        for m, freq in freq_table.items():
            p = freq_table[m] / n
            H += -p * np.log(p)
        
        return H / np.log(2)

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if not( epoch != 0 \
        and epoch % self.config["epoch_period"] == 0):
            return outputs_stream_dict

        if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
            experiences = input_streams_dict["experiences"].cpu().detach().numpy()
            representations = input_streams_dict["representations"].cpu().detach().numpy()
            latent_representations = input_streams_dict["latent_representations"].cpu().detach().numpy()
            if 'test' in mode:
                import ipdb; ipdb.set_trace()
                #TODO need to check that index are consistent with dataset index.
            indices = input_streams_dict["indices"].cpu().detach().numpy()
            
            for idx, tidx in enumerate(indices.tolist()):
                self.experiences[tidx] = experiences[idx]
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

        self.experiences = np.stack(self.experiences.values(), axis=0)
        self.representations = np.stack(self.representations.values(), axis=0)
        
        latent_shape = self.latent_representations.values()[-1].shape
        self.latent_representations = np.stack(self.latent_representations.values(), axis=0)
        self.indices = np.concatenate(self.indices, axis=0).reshape(-1)

        # Make sure every index is only seen once:
        self.indices, in_batch_indices = np.unique(self.indices, return_index=True)
        import ipdb; ipdb.set_trace()
        #TODO : check that values are ordered :S ?
        print(self.representations.values()[0] == self.representations[0])

        #self.representations = self.representations[in_batch_indices,:]
        #self.latent_representations = self.latent_representations[in_batch_indices,:]
        
        all_sentences = self.representations.values()
        sentence_length = all_sentences[0].shape[0]
        import ipdb; ipdb.set_trace()
        unique_sentences = set(all_sentences)
        
        per_unique_sentence_stats = {}
        previous_sentence = None
        compactness_count = 0
        for idx, sentence in enumerate(all_sentences):
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
        
        model = input_streams_dict["model"]
        training = model.training
        if hasattr(model, "eval"):  model.eval()
        
        mode = input_streams_dict["mode"]
        dataset = input_streams_dict["dataset"].datasets[mode]
        logger = input_streams_dict["logger"]

        columns = [f"idx"]
        columns += [f"token{idx}" for idx in range(sentence_length)]
        columns += ["stimulus"]
        columns += ["nbr_compact_segment"]
        columns += ["min_compactness", "max_compactness"]
        columns += [f"latent{idx}" for idx in range(latent_shape[1])]
        
        self.sample_table = wandb.Table(columns=columns) 
                    
        min_sum = 0
        max_sum = 0
        normalizer = len(all_sentences)
        for idx, sentence in enumerate(all_sentences):
            data = []

            data.append(idx)

            for widx in sentence:
                word = model.idx2w[widx.item()]
                data.append(word)

            exp = self.experiences[idx]
            nbr_frames = exp.shape[0] // 4
            stimulus_t = exp.reshape(nbr_frames,4,*exp.shape[-2:])[:,:3]*255
            stimulus_t = stimulus_t.astype(np.uint8)
            stimulus_t = wandb.Video(stimulus_t, fps=1, format="mp4")
            data.append(stimulus_t)
                
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

            self.sample_table.add_data(*data)
        
        wandb.log({f"{mode}/{self.id}/PerEpoch/CompactnessTable":self.sample_table}, commit=False)

        ## Compute Compactness Score:
        list_compactness_counts = []
        for us, stat in per_unique_sentence_stat.items():
            list_compactness_counts.append(*stat['compactness_counts'])
        import ipdb; ipdb.set_trace()
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
        
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Mean"] = mean_compactness_counts
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Std"] = std_compactness_counts
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Min"] = min(values)
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Max"] = max(values)
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Median"] = median_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Q1"] = q1_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Q3"] = q3_value
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/IQR"] = iqr
        
        mean_min_compactness = float(min_sum) / normalizer
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Minimal/Mean"] = mean_min_compactness
        mean_max_compactness =  float(max_sum) / normalizer
        logs_dict[f"{mode}/{self.id}/CompactnessCounts/Maximal/Mean"] = mean_max_compactness

        list_nbr_compact_segment = [len(ps['compactness_counts']) for ps in per_unique_sentence_stats.values()]
        mean_nbr_compact_segment = mean(list_nbr_compact_segment)
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
        
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Mean"] = mean_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Std"] = std_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Min"] = min_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Max"] = max_nbr_compact_segment
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Median"] = median_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Q1"] = q1_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/Q3"] = q3_value
        logs_dict[f"{mode}/{self.id}/NbrCompactSegments/IQR"] = iqr
        
        average_max_compactness_count = len(self.representations) / len(unique_sentences)
        threshold = max(1, int(0.75*average_max_compactness_count))
        nbr_max_compactness_count_greater_than_threshold = len([
            count for count in list_compactness_counts if count >= threshold]
        )
        compactness_score = float(nbr_max_compactness_count_greater_than_threshold) / len(list_compactness_counts)

        logs_dict[f"{mode}/{self.id}/CompactnessAmbiguityScore"] = compactness_score 

        self.experiences = {}
        self.representations = {}
        self.latent_representations = {}
        self.indices = []
        
        if hasattr(model, "eval"):  model.train(training)
                
        return outputs_stream_dict
 
