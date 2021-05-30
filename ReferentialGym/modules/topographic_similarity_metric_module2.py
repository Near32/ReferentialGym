from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import copy 

from ReferentialGym.modules import Module
from ReferentialGym.utils.utils import compute_topographic_similarity_parallel


def build_TopographicSimilarityMetricModule2(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return TopographicSimilarityMetricModule2(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class TopographicSimilarityMetricModule2(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None):

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

            "model":"modules:current_speaker:ref:ref_agent:",
            "features":"modules:current_speaker:ref:ref_agent:features",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
            "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
            "indices":"current_dataloader:sample:speaker_indices", 
            
        }
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(TopographicSimilarityMetricModule2, self).__init__(
            id=id,
            type="TopographicSimilarityMetric2",
            config=config,
            input_stream_ids=input_stream_ids)
        
        self.pvalue_significance_threshold = self.config["pvalue_significance_threshold"]
        self.whole_epoch_sentences = []
        self.max_workers = self.config["parallel_TS_computation_max_workers"]

        # Default = 0.0
        self.repr_dim_filtering_threshold = self.config["threshold"]
        
        current_random_state = np.random.get_state()
        np.random.seed(self.config['random_state_seed'])
        self.random_state = np.random.get_state()
        np.random.set_state(current_random_state)
        
        self.representations = []
        self.features = []
        self.latent_representations = []
        self.latent_representations_v = []
        self.latent_representations_ohe = []
        self.representations_indices = []
        self.indices = []

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]

    def _prune_dims(self, variances):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= self.repr_dim_filtering_threshold

    def _generate_training_batch(
        self,
        dataset,
        model,
        batch_size,
        nbr_points,
        active_dims):
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
        features = []
        latent_representations = []
        latent_representations_v = []
        latent_representations_ohe = []
        i = 0
        while i < nbr_points:
            num_points = min(nbr_points-i, batch_size)
            rep, feat, lrep, \
            lrep_v, lrep_ohe = self._generate_training_sample(
                dataset=dataset,
                model=model,
                batch_size=num_points, 
                active_dims=active_dims)
            # (batch_size, dim)
            representations.append(rep)
            features.append(feat)
            latent_representations.append(lrep)
            latent_representations_v.append(lrep_v)
            latent_representations_ohe.append(lrep_ohe)
            i+= num_points
        
        representations = np.concatenate(representations, axis=0)
        features = np.concatenate(features, axis=0)
        latent_representations = np.concatenate(latent_representations, axis=0)
        latent_representations_v = np.concatenate(latent_representations_v, axis=0)
        latent_representations_ohe = np.concatenate(latent_representations_ohe, axis=0)

        self.random_state = copy.deepcopy(np.random.get_state())
        np.random.set_state(current_random_state)
        
        #return np.transpose(representations), np.transpose(features), np.transpose(latent_representations)
        return representations, features, latent_representations, latent_representations_v, latent_representations_ohe

    def _generate_training_sample(self, 
                                  dataset, 
                                  model,
                                  batch_size, 
                                  active_dims):
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

            relevant_features = None
            if "preprocess_fn" in self.config:
                observations = self.config["preprocess_fn"](observations)
                relevant_features = observations

            output = model(observations)
            if "postprocess_fn" in self.config:
                relevant_representations = self.config["postprocess_fn"](output)
            else:
                relevant_representations = output
            
            if "features_postprocess_fn" in self.config:
                relevant_features = self.config["features_postprocess_fn"](output)
            elif relevant_features is None:
                relevant_features = output

            factors_v = dataset.sample_latents_values_from_factors(
              factors, 
              random_state=None, #self.random_state
            )

            factors_ohe = dataset.sample_latents_ohe_from_factors(
              factors, 
              random_state=None, #self.random_state
            )

            relevant_latent_representations = factors
            relevant_latent_representations_v = factors_v
            relevant_latent_representations_ohe = factors_ohe
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
            relevant_features = self.features[relevant_samples_indices_sampled]
            # (batch_size, feat_dim)
            relevant_representations = self.representations[relevant_samples_indices_sampled]
            # (batch_size, repr_dim)
            relevant_latent_representations = self.latent_representations[relevant_samples_indices_sampled]
            # (batch_size, latent_repr_dim)
            relevant_latent_representations_v = self.latent_representations_v[relevant_samples_indices_sampled]
            # (batch_size, latent_v_repr_dim)
            relevant_latent_representations_ohe = self.latent_representations_ohe[relevant_samples_indices_sampled]
            # (batch_size, latent_ohe_repr_dim)

        return relevant_representations, relevant_features, relevant_latent_representations, relevant_latent_representations_v, relevant_latent_representations_ohe

    def _compute_topo_sim(self, np_sentences, np_features, comprange=None) -> Tuple[float, float, float]:
        _, idx_unique_sentences = np.unique(np_sentences, axis=0, return_index=True)
        
        toposim, p, levs, cossims = compute_topographic_similarity_parallel(
            sentences=np_sentences, 
            features=np_features, 
            comprange=comprange,
            max_workers=self.max_workers
        )

        unique_prod_ratio = len(idx_unique_sentences) / len(np_sentences) * 100.0

        return toposim, p, unique_prod_ratio
    
    def _compute_global_variances(self, model:object, dataset:object, nbr_points:int, batch_size:int=64) -> Tuple[object, object]:
        if self.config['resample']:
            latent_representations = []
            representations = []
            for _ in range((nbr_points//batch_size)+1):
                # Sample two mini batches of latent variables.
                factors = dataset.sample_factors(
                    batch_size, 
                    random_state=None, #self.random_state
                )

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

                latent_representations.append(factors)
                representations.append(relevant_representations)

            representations = np.vstack(representations)
            latent_representations = np.vstack(latent_representations)
        else:
            representations = self.representations
            latent_representations = self.latent_representations
        
        global_variances = np.var(representations, axis=0, ddof=1)
        latent_global_variances = np.var(latent_representations, axis=0, ddof=1)

        self.nbr_factors = latent_representations.shape[-1]

        return global_variances, latent_global_variances

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        epoch = input_streams_dict["epoch"]        
        mode = input_streams_dict["mode"]
        
        if epoch % self.config["epoch_period"] == 0 and "train" in mode:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                representations = input_streams_dict["representations"]
                while representations.shape[-1] == 1:
                    representations = representations.squeeze(-1)
                self.representations.append(representations.cpu().detach().numpy())
                features = input_streams_dict["features"]
                while features.shape[-1] == 1:
                    features = features.squeeze(-1)
                self.features.append(features.cpu().detach().numpy())
                latent_representations = input_streams_dict["latent_representations"]
                self.latent_representations.append(latent_representations.cpu().detach().numpy())
                latent_representations_v = input_streams_dict["latent_representations_values"]
                self.latent_representations_v.append(latent_representations_v.cpu().detach().numpy())
                latent_representations_ohe = input_streams_dict["latent_representations_one_hot_encoded"]
                self.latent_representations_ohe.append(latent_representations_ohe.cpu().detach().numpy())
                indices = input_streams_dict["indices"]
                self.indices.append(indices.cpu().detach().numpy())

            # Is it the end of the epoch?
            end_of_epoch = all([
              input_streams_dict[key]
              for key in self.end_of_]
            )
            
            not_empty = len(self.indices) > 0
            
            if end_of_epoch and (not_empty or self.config["resample"]):
                if not_empty:
                    repr_last_dim = self.representations[-1].shape[-1] 
                    self.representations = np.concatenate(self.representations, axis=0).reshape(-1, repr_last_dim)
                    feat_last_dim = self.features[-1].shape[-1] 
                    self.features = np.concatenate(self.features, axis=0).reshape(-1, feat_last_dim)
                    latent_repr_last_dim = self.latent_representations[-1].shape[-1] 
                    self.latent_representations = np.concatenate(self.latent_representations, axis=0).reshape(-1, latent_repr_last_dim)
                    
                    latent_v_repr_last_dim = self.latent_representations_v[-1].shape[-1] 
                    self.latent_representations_v = np.concatenate(self.latent_representations_v, axis=0).reshape(-1, latent_v_repr_last_dim)
                    
                    latent_ohe_repr_last_dim = self.latent_representations_ohe[-1].shape[-1] 
                    self.latent_representations_ohe = np.concatenate(self.latent_representations_ohe, axis=0).reshape(-1, latent_ohe_repr_last_dim)
                    
                    self.indices = np.concatenate(self.indices, axis=0).reshape(-1)

                    # Make sure every index is only seen once:
                    self.indices, in_batch_indices = np.unique(self.indices, return_index=True)
                    self.features = self.features[in_batch_indices,:]
                    self.representations = self.representations[in_batch_indices,:]
                    self.latent_representations = self.latent_representations[in_batch_indices,:]
                    self.latent_representations_v = self.latent_representations_v[in_batch_indices,:]
                    self.latent_representations_ohe = self.latent_representations_ohe[in_batch_indices,:]
                    
                model = input_streams_dict["model"]
                if hasattr(model, "eval"):  model.eval()
                
                mode = input_streams_dict["mode"]
                dataset = input_streams_dict["dataset"].datasets[mode]
                logger = input_streams_dict["logger"]

                global_variances, latent_global_variances = self._compute_global_variances(
                    model=model,
                    dataset=dataset,
                    batch_size=self.config["batch_size"],
                    nbr_points=self.config["nbr_train_points"],
                )

                active_dims = self._prune_dims(global_variances)
                self.active_latent_dims = [idx for idx, var in enumerate(latent_global_variances)
                                            if var > 0.0]
                scores_dict = {}

                scores_dict["topo_sims"] = 0.
                scores_dict["topo_sims_ohe"] = 0.
                scores_dict["topo_sims_v"] = 0.
                scores_dict["feat_topo_sims"] = 0.

                scores_dict["num_active_dims"] = 0

                pvalues = 1.0
                pvalues_v = 1.0
                pvalues_ohe = 1.0
                pvalues_feat = 1.0 

                unique_prod_ratios = 0.0
                comprange = None
                if self.config.get("metric_fast", False):
                    comprange = 0.2 # 20%

                if active_dims.any():
                    train_repr, train_feat, train_lrepr, \
                    train_lrepr_v, train_lrepr_ohe = self._generate_training_batch(
                        dataset=dataset,
                        model=model, 
                        batch_size=self.config["batch_size"],
                        nbr_points=self.config["nbr_train_points"], 
                        active_dims=active_dims)
                    # (nbr_points, dim)
                    
                    try:
                        scores_dict["topo_sims"], pvalues, unique_prod_ratios = self._compute_topo_sim(
                            np_sentences=train_repr, 
                            np_features=train_lrepr, 
                            comprange=comprange,
                        )
                    except Exception as e:
                        print(f"TOPOSIM :: exception caught: {e}")

                    try:
                        scores_dict["topo_sims_v"], pvalues_v, unique_prod_ratios_v = self._compute_topo_sim(
                            np_sentences=train_repr, 
                            np_features=train_lrepr_v, 
                            comprange=comprange,
                        )
                    except Exception as e:
                        print(f"TOPOSIM Values :: exception caught: {e}")

                    try:
                        scores_dict["topo_sims_ohe"], pvalues_ohe, unique_prod_ratios_ohe = self._compute_topo_sim(
                            np_sentences=train_repr, 
                            np_features=train_lrepr_ohe, 
                            comprange=comprange,
                        )
                    except Exception as e:
                        print(f"TOPOSIM OHE :: exception caught: {e}")

                    try:
                        scores_dict["feat_topo_sims"], pvalues_feat, unique_prod_ratios_feat = self._compute_topo_sim(
                            np_sentences=train_repr, 
                            np_features=train_feat, 
                            comprange=comprange,
                        )
                    except Exception as e:
                        print(f"TOPOSIM Feat :: exception caught: {e}")
                    
                    scores_dict["num_active_dims"] = len(active_dims)
                    

                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity/TopoSim"] = scores_dict["topo_sims"]*100.0
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity/NonAmbiguousProduction"] = unique_prod_ratios
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity/PValues"] = pvalues
                
                if pvalues < self.pvalue_significance_threshold:
                    logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity/TopoSim/significant"] = scores_dict["topo_sims"]*100.0
                
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_OHE/TopoSim"] = scores_dict["topo_sims_ohe"]*100.0
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_OHE/PValues"] = pvalues_ohe
                
                if pvalues_ohe < self.pvalue_significance_threshold:
                    logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_OHE/TopoSim/significant"] = scores_dict["topo_sims_ohe"]*100.0
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Values/TopoSim"] = scores_dict["topo_sims_v"]*100.0
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Values/PValues"] = pvalues_v
                
                if pvalues_v < self.pvalue_significance_threshold:
                    logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Values/TopoSim/significant"] = scores_dict["topo_sims_v"]*100.0
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Features/TopoSim"] = scores_dict["feat_topo_sims"]*100.0
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Features/PValues"] = pvalues_feat
                
                if pvalues_feat < self.pvalue_significance_threshold:
                    logs_dict[f"{mode}/{self.id}/CompositionalityMetric/TopographicSimilarity_Features/TopoSim/significant"] = scores_dict["feat_topo_sims"]*100.0
                
                self.representations = []
                self.features = []
                self.latent_representations = []
                self.latent_representations_v = []
                self.latent_representations_ohe = []
                self.representations_indices = []
                self.indices = []

                if hasattr(model, "eval"):  model.train()
                

        return outputs_stream_dict
    