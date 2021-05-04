from typing import Dict, List, Tuple

import numpy as np 
import sklearn  
import copy

from .module import Module

eps = 1e-12

def build_PositionalBagOfSymbolsDisentanglementMetricModule(id:str,
                               config:Dict[str,object],
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return PositionalBagOfSymbolsDisentanglementMetricModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class PositionalBagOfSymbolsDisentanglementMetricModule(Module):
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

            "model":"modules:current_speaker:ref:ref_agent:",
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

        super(PositionalBagOfSymbolsDisentanglementMetricModule, self).__init__(id=id,
                                                 type="PositionalBagOfSymbolsDisentanglementMetricModule",
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
        
        self.nbr_factors = self.latent_representations.shape[-1]
        
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

    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L15
    def entropy_dict(self, freq_table):
        H = 0
        n = sum(v for v in freq_table.values())

        for m, freq in freq_table.items():
            p = freq_table[m] / n
            H += -p * np.log(p)
        
        return H / np.log(2)

    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L25
    def _compute_entropy2(self, messages):
        from collections import defaultdict

        freq_table = defaultdict(float)

        for m in messages:
            m = self._hashable_tensor(m)
            freq_table[m] += 1.0

        return self.entropy_dict(freq_table)

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

    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L50
    def _compute_mutual_info2(self, xs, ys):
        e_x = self._compute_entropy2(xs)
        e_y = self._compute_entropy2(ys)

        xys = []

        for x, y in zip(xs, ys):
            xy = (self._hashable_tensor(x), self._hashable_tensor(y))
            xys.append(xy)

        e_xy = self._compute_entropy2(xys)

        return e_x + e_y - e_xy

    # Adapted from: 
    # https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/compo_vs_generalization/intervention.py#L48
    def _compute_score_chaabouni(self, meanings, representations):
        global eps
        gaps = np.zeros(representations.shape[1])
        non_constant_positions = 0.0

        for j in range(representations.shape[1]):
            symbol_mi = []
            h_j = None
            for i in range(meanings.shape[1]):
                x, y = meanings[:, i], representations[:, j]
                info = self._compute_mutual_info2(x, y)
                symbol_mi.append(info)

                if h_j is None:
                    h_j = self._compute_entropy2(y)

            symbol_mi.sort(reverse=True)

            if h_j > 0.0:
                gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
                non_constant_positions += 1

        if non_constant_positions==0:
            score = -1
        else:
            score = (gaps.sum() / (non_constant_positions+eps)).item()
        return score

    def _compute_score_denamganai(self, mi, rent, lent):
        # (rep_dim x lrep_dim)
        # sorting from max:0 to min...:
        """
        sorted_mi = np.sort(mi, axis=1)[:, ::-1]
        # (rep_dim x lrep_dim)
        
        # ent: # (rep_dim,)
        
        score = np.divide(
            sorted_mi[:, 0] - sorted_mi[:, 1], 
            rent[:]
        )
        
        indicator = (rent>0)
        score = score * indicator
        s1 = score.sum() / indicator.sum() 
        """

        # Replication of MIG approach:
        sorted_mi = np.sort(mi, axis=0)[::-1]
        # (rep_dim x lrep_dim)
        
        # ent: # (rep_dim,)
        
        """
        score = np.divide(
            sorted_mi[0, :] - sorted_mi[1, :], 
            lent[:]
        )
        
        indicator = (lent>0)
        score = score * indicator
        """
        score = []
        lent = lent[:]
        nbr_nonnul_entry = 0
        for idx in range(sorted_mi.shape[1]):
            if lent[idx]>0:
                nbr_nonnul_entry += 1
                sidx = (sorted_mi[0, idx] - sorted_mi[1, idx])/lent[idx]
                score.append(sidx)
        if nbr_nonnul_entry == 0:
            s12 = -1
        else:
            s12 = sum(score) / nbr_nonnul_entry 

        # Replication of 2:
        """
        rep_dim = mi.shape[0]
        gaps = [] #np.zeros(rep_dim)
        for rdidx in range(rep_dim):
            if rent[rdidx] > 0.0:
                smi_over_ldim = np.sort(mi[rdidx], axis=0)[::-1]
                gaps.append(
                    (smi_over_ldim[0]-smi_over_ldim[1]) / rent[rdidx]
                )

        s2 = sum(gaps)/len(gaps)
        """

        return s12

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

    def _bos_repr(self, rep):
        sentence_length = rep.shape[0]
        nbr_samples = rep.shape[1]
        eff_vocab_size = int(rep.max())+1 # counting elements...

        bos_repr = np.zeros((eff_vocab_size, nbr_samples), dtype=np.int32)

        # rep: (rep_dim, batch_size)
        for v in range(eff_vocab_size):
            bos_repr[v, :] = (rep==v).sum(axis=0) #summing over the sentence dimension (not batch)

        return bos_repr

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

        return global_variances, latent_global_variances
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        if epoch % self.config["epoch_period"] == 0:
            if self.config.get("filtering_fn", (lambda x: True))(input_streams_dict):
                representations = input_streams_dict["representations"]
                while representations.shape[-1] == 1:
                    representations = representations.squeeze(-1)
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
            
            if end_of_epoch and not_empty:
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

                if not active_dims.any():
                    scores_dict["discr_posdis_score_denamganai"] = 0.
                    scores_dict["posdis_score_denamganai"] = 0.
                    scores_dict["posdis_score_chaabouni"] = 0.

                    scores_dict["num_active_dims"] = 0
                    
                    scores_dict["discr_bosdis_score_denamganai"] = 0.
                    scores_dict["bosdis_score_denamganai"] = 0.
                    scores_dict["bosdis_score_chaabouni"] = 0.
                else:
                    train_repr, train_lrepr = self._generate_training_batch(
                        dataset=dataset,
                        model=model, 
                        batch_size=self.config["batch_size"],
                        nbr_points=self.config["nbr_train_points"], 
                        active_dims=active_dims)
                    # (dim, nbr_points)
                    
                    ###########################################
                    ###########################################
                    # Positional Disentanglement
                    ###########################################
                    # Discretization:
                    discr_train_repr = self._discretize(train_repr, num_bins=20)

                    discr_mutual_information = self._compute_mutual_info(discr_train_repr, train_lrepr) 
                    mutual_information = self._compute_mutual_info(train_repr, train_lrepr) 
                    # (rep_dim, lrep_dim)

                    entropy = self._compute_entropy(train_repr)
                    # (rep_dim,)
                    lentropy = self._compute_entropy(train_lrepr)
                    # (lrep_dim,)
                    
                    dms_d = self._compute_score_denamganai(discr_mutual_information, entropy, lentropy)
                    ms_d = self._compute_score_denamganai(mutual_information, entropy, lentropy)

                    ms_c = self._compute_score_chaabouni(meanings=train_lrepr.transpose(), representations=train_repr.transpose())

                    scores_dict["num_active_dims"] = len(active_dims)
                    
                    scores_dict["discr_posdis_score_denamganai"] = dms_d
                    scores_dict["posdis_score_denamganai"] = ms_d
                    scores_dict["posdis_score_chaabouni"] = ms_c 

                    # To what extent is a factor captured in a modular way by the model?
                    per_factor_maxmi = np.max(mutual_information, axis=0)

                    for idx, maxmi in enumerate(per_factor_maxmi):
                        logs_dict[f"{mode}/{self.id}/CompositionalityMetric/PositionalDisentanglement/MaxMutualInformation/factor_{idx}"] = maxmi
                    
                    ###########################################
                    ###########################################
                    # Bag-of-symbols Disentanglement:
                    ###########################################                    
                    bos_train_repr = self._bos_repr(train_repr)
                    # (vocab_size(+1), nbr_points)

                    # Dropping EoS symbol?
                    #bos_train_repr = bos_train_repr[1:, :]

                    bos_entropy = self._compute_entropy(bos_train_repr)
                    
                    # Discretization:
                    discr_bos_train_repr = self._discretize(bos_train_repr, num_bins=20)

                    bos_discr_mutual_information = self._compute_mutual_info(discr_bos_train_repr, train_lrepr) 
                    bos_mutual_information = self._compute_mutual_info(bos_train_repr, train_lrepr) 
                    # (rep_dim, lrep_dim)
                    
                    try:
                        bos_dms_d = self._compute_score_denamganai(bos_discr_mutual_information, bos_entropy, lentropy)
                    except Exception as e:
                        bos_dms_d = 0.0
                        print(f"DISCR BOSDIS Denamganai :: exception caught: {e}")
                    
                    try:
                        bos_ms_d = self._compute_score_denamganai(bos_mutual_information, bos_entropy, lentropy)
                    except Exception as e:
                        bos_ms_d = 0.0
                        print(f"BOSDIS Denamganai :: exception caught: {e}")
                    
                    try:
                        bos_ms_c = self._compute_score_chaabouni(meanings=train_lrepr.transpose(), representations=bos_train_repr.transpose())
                    except Exception as e:
                        bos_ms_c = 0.0
                        print(f"BOSDIS Chaabouni :: exception caught: {e}")
                    
                    scores_dict["discr_bosdis_score_denamganai"] = bos_dms_d
                    scores_dict["bosdis_score_denamganai"] = bos_ms_d
                    scores_dict["bosdis_score_chaabouni"] = bos_ms_c
                    
                    # To what extent is a factor captured in a modular way by the model?
                    per_factor_bosmaxmi = np.max(bos_mutual_information, axis=0)

                    for idx, maxmi in enumerate(per_factor_bosmaxmi):
                        logs_dict[f"{mode}/{self.id}/CompositionalityMetric/BagOfSymbolsDisentanglement/MaxMutualInformation/factor_{idx}"] = maxmi
                    
                    ###########################################
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/PositionalDisentanglement/PosDisScore/Discretized"] = scores_dict["discr_posdis_score_denamganai"]
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/PositionalDisentanglement/PosDisScore/SpeakerCentred"] = scores_dict["posdis_score_denamganai"]
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/PositionalDisentanglement/PosDisScore/ListenerCentred"] = scores_dict["posdis_score_chaabouni"]
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/BagOfSymbolsDisentanglement/BosDisScore/SpeakerCentred/Discretized"] = scores_dict["discr_bosdis_score_denamganai"]
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/BagOfSymbolsDisentanglement/BosDisScore/SpeakerCentred/"] = scores_dict["bosdis_score_denamganai"]
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/BagOfSymbolsDisentanglement/BosDisScore/ListenerCentred/"] = scores_dict["bosdis_score_chaabouni"]
                
                logs_dict[f"{mode}/{self.id}/CompositionalityMetric/PositionalBagOfSymbolsDisentanglement/nbr_active_dims"] = scores_dict["num_active_dims"]

                self.representations = []
                self.latent_representations = []
                self.representations_indices = []
                self.indices = []
                
                if hasattr(model, "eval"):  model.train()
                
        return outputs_stream_dict
    
