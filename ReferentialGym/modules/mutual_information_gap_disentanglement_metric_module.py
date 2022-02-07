from typing import Dict, List, Tuple

import numpy as np 
import sklearn
import copy

from .module import Module

"""
Based on:
https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/mig.py
"""

eps = 1e-8


def build_MutualInformationGapDisentanglementMetricModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return MutualInformationGapDisentanglementMetricModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class MutualInformationGapDisentanglementMetricModule(Module):
    def __init__(self,
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

            "model":"modules:current_speaker:ref:ref_agent:cnn_encoder",
            "representations":"modules:current_speaker:ref:ref_agent:features",
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

        super(MutualInformationGapDisentanglementMetricModule, self).__init__(id=id,
                                                 type="MutualInformationGapDisentanglementMetricModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        # Default = 0.0
        self.repr_dim_filtering_threshold = self.config["threshold"]
        
        current_random_state = np.random.get_state()
        np.random.seed(self.config['random_state_seed'])
        self.random_state = np.random.get_state()
        np.random.set_state(current_random_state)

        self.representations = []
        self.latent_representations = []
        self.representations_indices = []
        self.indices = []

        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
    
    def _prune_dims(self, variances):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= self.repr_dim_filtering_threshold

    def _generate_training_batch(self,
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

    def _compute_mutual_info(self, rep, lrep):
        rep_size = rep.shape[0]
        lrep_size = lrep.shape[0]
        mi = np.zeros([rep_size, lrep_size])
        for i in range(rep_size):
            for j in range(lrep_size):
                mi[i, j] = sklearn.metrics.mutual_info_score(lrep[j, :], rep[i, :])
        return mi

    def _compute_entropy(self, rep):
        rep_size = rep.shape[0]
        ent = np.zeros([rep_size,])
        for i in range(rep_size):
            ent[i] = sklearn.metrics.mutual_info_score(rep[i, :], rep[i, :])
        return ent

    def _compute_mutual_information_gap_score(self, mi, ent):
        global eps
        # ent: # (lrep_dim, )
        # mi: # (rep_dim x lrep_dim)
        # sorting from max:0 to min...:
        sorted_mi = np.sort(mi, axis=0)[::-1]
        # (rep_dim x lrep_dim)
        
        mig_scores = np.divide(
            sorted_mi[0, :] - sorted_mi[1, :], 
            ent[:]+eps
        )
        
        s1 = np.mean(mig_scores)

        nonnullent_mig_scores = (-1)*np.ones(ent.shape)
        sum_nonnullent_mig_scores = 0
        nbr_nonnullent_dim = 0
        for idx, enti in enumerate(ent[:]):
            if enti > eps:
                nbr_nonnullent_dim += 1
                nonnullent_mig_scores[idx] = np.divide(
                    sorted_mi[0, idx] - sorted_mi[1, idx], 
                    enti
                )
                sum_nonnullent_mig_scores += nonnullent_mig_scores[idx]

        nne_s = sum_nonnullent_mig_scores/nbr_nonnullent_dim

        """
        sorted_mi = np.sort(mi, axis=1)[:, ::-1]
        # (rep_dim x lrep_dim)
        # ent: # (rep_dim,)
        
        score = np.divide(
            sorted_mi[:, 0] - sorted_mi[:, 1], 
            ent[:]
        )
        
        s2 = np.mean(score)
        """

        return s1, mig_scores, nne_s, nonnullent_mig_scores

    def _discretize(self, target, num_bins=20):
        """
        Discretization based on histograms.
        """
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(
                target[i, :], 
                np.histogram(
                    target[i, :], 
                    num_bins
                )[1][:-1]
            )
        return discretized

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
        global eps 
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if epoch != 0 \
        and epoch % self.config["epoch_period"] == 0 and "train" in mode:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                representations = input_streams_dict["representations"]
                self.representations.append(representations.cpu().detach().numpy())
                latent_representations = input_streams_dict["latent_representations"]
                self.latent_representations.append(latent_representations.cpu().detach().numpy())
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
                    latent_repr_last_dim = self.latent_representations[-1].shape[-1] 
                    self.latent_representations = np.concatenate(self.latent_representations, axis=0).reshape(-1, latent_repr_last_dim)
                    self.indices = np.concatenate(self.indices, axis=0).reshape(-1)

                    # Make sure every index is only seen once:
                    self.indices, in_batch_indices = np.unique(self.indices, return_index=True)
                    self.representations = self.representations[in_batch_indices,:]
                    self.latent_representations = self.latent_representations[in_batch_indices,:]
                    
                model = input_streams_dict["model"]
                model.eval()
                
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

                if not active_dims.any():
                    scores_dict["mig_score"] = 0.
                    scores_dict["nne_mig_score"] = 0.
                    #scores_dict["nondiscr_mig_score"] = 0.
                    scores_dict["eval_accuracy"] = 0.
                    scores_dict["num_active_dims"] = 0
                else:
                    train_repr, train_lrepr = self._generate_training_batch(
                        dataset=dataset,
                        model=model, 
                        batch_size=self.config["batch_size"],
                        nbr_points=self.config["nbr_train_points"], 
                        active_dims=active_dims)
                    # (dim, nbr_points)
                    """
                    test_repr, test_lrepr = self._generate_training_batch(
                        dataset=dataset,
                        model=model, 
                        batch_size=self.config["batch_size"],
                        nbr_points=self.config["nbr_eval_points"],
                        active_dims=active_dims
                    )
                    # (dim, nbr_points)
                    """
                    # Discretization: necessary!
                    discr_train_repr = self._discretize(train_repr, num_bins=20)

                    mutual_information = self._compute_mutual_info(discr_train_repr, train_lrepr) 
                    #nondiscr_mutual_information = self._compute_mutual_info(train_repr, train_lrepr) 
                    # (rep_dim, lrep_dim)
                    entropy = self._compute_entropy(train_lrepr)
                    # (lrep_dim,)

                    ms, mig_scores, nne_ms, nne_mig_scores = self._compute_mutual_information_gap_score(
                        mutual_information, 
                        entropy
                    )
                    #ndms, nondiscr_mig_scores = self._compute_mutual_information_gap_score(nondiscr_mutual_information, entropy)

                    scores_dict["mig_score"] = ms
                    scores_dict["nne_mig_score"] = nne_ms
                    #scores_dict["nondiscr_mig_score"] = ndms
                    scores_dict["num_active_dims"] = len(active_dims)
                    
                    for idx, msi in enumerate(mig_scores[:]):
                        logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/NormalizedMutualInformationGap/factor_{idx}"] = msi
                    
                    for idx, nne_msi in enumerate(nne_mig_scores[:]):
                        logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/NonNullEntropy-NormalizedMutualInformationGap/factor_{idx}"] = nne_msi
                                        
                    # To what extent is a factor captured in a modular way by the model?
                    per_factor_maxmi = np.max(mutual_information, axis=0)

                    for idx, maxmi in enumerate(per_factor_maxmi):
                        logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/MaxMutualInformation/factor_{idx}"] = maxmi
                        nmmi = -1
                        entidx = entropy[:][idx]
                        if entidx > eps:
                            nmmi = maxmi/entidx
                        logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/NormalizedMaxMutualInformation/factor_{idx}"] = nmmi
                    
                    for idx, enti in enumerate(entropy[:]):
                        logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/Entropy/factor_{idx}"] = enti
                    
                logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/MIGScore"] = scores_dict["mig_score"]
                logs_dict[f"{mode}/{self.id}/DisentanglementMetric/NonNullEntropy-MutualInformationGap/MIGScore"] = scores_dict["nne_mig_score"]
                #logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/NonDiscrMIGScore"] = scores_dict["nondiscr_mig_score"]
                logs_dict[f"{mode}/{self.id}/DisentanglementMetric/MutualInformationGap/nbr_active_dims"] = scores_dict["num_active_dims"]

                self.representations = []
                self.latent_representations = []
                self.representations_indices = []
                self.indices = []

                model.train()
                
            
        return outputs_stream_dict
    
