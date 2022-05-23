from typing import Dict, List 

import torch
import numpy as np
import copy 

from ..modules import Module
from ..datasets import shuffle, collate_dict_wrapper

exclude = True 

class OracleDatasamplingModule(Module):
    def __init__(self, 
                 id:str,
                 config:Dict[str,object]):
        """
        :param id: str defining the ID of the module.
        :param config: Dict of hyperparameters:
            - "operating_mode": str defining the mode in which this module operates.
            - "dataset_key": str key of the dataset to use out of the current DualLabeledDataset.
            - "batch_size": int batch size to sample with.
            - "nbr_attribute": int defining the number of attributes in the dataset.
            - "sample_type_likelihood": Dict whose keys are integers indices to the attributes
                that should be considered varying when sampling. The values correspond to 
                (not necessarily normalized) likelihoods of sampling observations whose
                corresponding attribute varies from speaker experience to listener experience.
                -1: key corresponding to speaker and listener experiences being the same.
                `len(dataset.attributes)`: key corresponding to completely random experiences.
                N.B.: default sampling likelihood puts all the weight on random experiences.
        """

        input_stream_ids = {
            "dataset":"current_dataset:ref",
            "epoch":"signals:epoch",
            "mode":"signals:mode",
            "use_cuda":"signals:use_cuda",
            "it_sample":"signals:it_sample",
            # step in the sequence of repetitions of the current batch
            "it_step":"signals:it_step",
            # step in the communication round.
            "it_round":"signals:obverter_round_iteration",
            # current round iteration.
            "sample":"current_dataloader:sample",
        }

        super(OracleDatasamplingModule, self).__init__(
            id=id, 
            type="OracleDatasamplingModule",
            config=config,
            input_stream_ids=input_stream_ids)

        self.operating_mode = self.config["operating_mode"]
        self.dataset_key = self.config["dataset_key"]
        self.batch_size = self.config["batch_size"]
        self.collate_fn = collate_dict_wrapper
        self.counterRounds = 0
        self.current_round_batches = []
        self.unloading = False

        self.sample_type_ranges = {}
        # Normalize sample_type_likelihood:
        sum_lklh = sum(self.config["sample_type_likelihood"].values())
        ## Adding default random:
        if self.config["nbr_attribute"] not in self.config["sample_type_likelihood"].keys():
            self.config["sample_type_likelihood"][self.config["nbr_attribute"] = max(0.0, 1.0-sum_lklh)
        if sum_lklh!=0:
            for k,v in self.config["sample_type_likelihood"].items():
                self.config["sample_type_likelihood"][k] = v/sum_lklh
        # Register sample_type_likelihood as random sampling ranges:
        range_max = 0.0 
        for k,v in self.config["sample_type_likelihood"].items():
            range_max += v
            self.sample_type_range[k] = range_max 

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        
        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_logits'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits over symbols.
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sentences_one_hot'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
            - `'exp_latents'`: Tensor of shape `(batch_size, nbr_latent_dimensions)`.
            - `'multi_round'`: Boolean defining whether to utter a sentence back or not.
            - `'graphtype'`: String defining the type of symbols used in the output sentence:
                        - `'categorical'`: one-hot-encoded symbols.
                        - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                        - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                        - `'obverter'`: obverter training scheme...
            - `'tau0'`: Float, temperature with which to apply gumbel-softmax estimator. 
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'config'`: Dict of hyperparameters to the referential game.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
            - `'it'`: Integer specifying the iteration number of the current function call.
        """
        global exclude 

        outputs_dict = {}


        epoch = input_streams_dict["epoch"]
        mode = input_streams_dict["mode"]
        it_step = input_streams_dict["it_step"]
        it_sample = input_streams_dict["it_sample"]
        counterRounds = input_streams_dict["it_round"]

        newRound = False 
        if counterRounds != self.counterRounds:
            newRound = True 
            self.counterRounds = counterRounds

        if newRound:
            self.unloading = not (self.unloading)
            size = len(self.current_round_batches) 
            assert size==0 or size==self.config["obverter_nbr_games_per_round"]


        if self.operating_mode in mode and it_step == 0:
            if self.unloading:
                assert len(self.current_round_batches)>0
                new_sample = self.current_round_batches.pop(0)
                outputs_dict["current_dataloader:sample"] = new_sample
                return outputs_dict

            if self.config.get("round_alternation_only", False):
                sample = input_streams_dict["sample"]
                self.current_round_batches.append(sample)
                return outputs_dict

            dataset = input_streams_dict["dataset"]
            # assumes DualLabeledDataset...
            operating_dataset = dataset.datasets[self.dataset_key]

            # Make the descriptive ratio no longer effective:
            dataset.kwargs["descriptive"] = False 

            batch = []
            for bidx in range(self.batch_size):
                sample_type_rnd = np.random.rand()
                stidx = -1
                while sample_type_rnd > self.sample_type_ranges[stidx]:
                    stidx += 1
                    while stidx not in self.sample_type_ranges.keys():
                        stidx +=1 

                if stdix==-1:
                    speaker_idx = np.random.randint(len(dataset))
                    latents_class = operating_dataset.getlatentclass(speaker_idx) 
                    color_id = latents_class[0]
                    shape_id = latents_class[1]
                    listener_idx = np.random.choice(
                        [
                            idx 
                            for idx in latents_to_possible_indices[color_id][shape_id]
                            if idx != speaker_idx
                        ]
                    )
                    batch.append(
                        self.sample(
                            dataset=dataset, 
                            speaker_idx=speaker_idx, 
                            listener_idx=listener_idx, 
                            same=True
                        )
                    )

            for i in range(n_same_shape):
                speaker_idx = np.random.randint(len(dataset))
                latents_class = train_dataset.getlatentclass(speaker_idx) 
                speaker_color_id = latents_class[0]
                shape_id = latents_class[1]
                choice_set = copy.deepcopy(train_dataset.same_shape_indices[shape_id])
                choice_set.remove(speaker_idx)
                
                # remove the speaker color:
                if exclude:
                    for idx in train_dataset.same_color_indices[speaker_color_id]:
                        if idx in choice_set:   choice_set.remove(idx)
                
                listener_idx = np.random.choice(choice_set)
                listener_color_id= train_dataset.getlatentclass(listener_idx)[0]
                same = (speaker_color_id == listener_color_id)

                batch.append(
                    self.sample(
                        dataset=dataset, 
                        speaker_idx=speaker_idx, 
                        listener_idx=listener_idx, 
                        same=same,
                    )
                )

            for i in range(n_same_color):
                speaker_idx = np.random.randint(len(dataset))
                latents_class = train_dataset.getlatentclass(speaker_idx) 
                color_id = latents_class[0]
                speaker_shape_id = latents_class[1]
                choice_set = copy.deepcopy(train_dataset.same_color_indices[color_id])
                choice_set.remove(speaker_idx)
                
                # remove the speaker shape:
                if exclude:
                    for idx in train_dataset.same_shape_indices[speaker_shape_id]:
                        if idx in choice_set:   choice_set.remove(idx)
                
                listener_idx = np.random.choice(choice_set)
                listener_shape_id= train_dataset.getlatentclass(listener_idx)[1]
                same = (speaker_shape_id == listener_shape_id)

                batch.append(
                    self.sample(
                        dataset=dataset, 
                        speaker_idx=speaker_idx, 
                        listener_idx=listener_idx, 
                        same=same,
                    )
                )

            for i in range(n_random):
                speaker_idx = np.random.randint(len(dataset))
                speaker_latents_class = train_dataset.getlatentclass(speaker_idx) 
                speaker_color_id = speaker_latents_class[0]
                speaker_shape_id = speaker_latents_class[1]
                
                listener_idx = np.random.randint(len(dataset))
                listener_latents_class = train_dataset.getlatentclass(listener_idx) 
                listener_color_id = listener_latents_class[0]
                listener_shape_id = listener_latents_class[1]
                
                same = (speaker_shape_id == listener_shape_id) and (speaker_color_id == listener_color_id)
                
                batch.append(
                    self.sample(
                        dataset=dataset, 
                        speaker_idx=speaker_idx, 
                        listener_idx=listener_idx, 
                        same=same
                    )
                )

            new_sample = self.collate_fn(batch)
            
            if input_streams_dict["use_cuda"]:
                new_sample = new_sample.cuda()

            outputs_dict["current_dataloader:sample"] = new_sample
            self.current_round_batches.append(new_sample)

        return outputs_dict

    def sample(self, dataset, speaker_idx, listener_idx, same:bool=True):
        # Creating speaker's dictionnary:
        speaker_sample_d = dataset.sample(idx=speaker_idx)
        
        # Adding batch dimension:
        for k,v in speaker_sample_d.items():
            if not(isinstance(v, torch.Tensor)):    
                v = torch.Tensor(v)
            speaker_sample_d[k] = v.unsqueeze(0)

        if dataset.kwargs['observability'] == "partial":
            for k,v in speaker_sample_d.items():
                speaker_sample_d[k] = v[:,0].unsqueeze(1)
        
        ##--------------------------------------------------------------
        ##--------------------------------------------------------------

        # Creating listener's dictionnary:
        listener_sample_d = dataset.sample(idx=listener_idx)
        
        # Adding batch dimension:
        for k,v in listener_sample_d.items():
            if not(isinstance(v, torch.Tensor)):    
                v = torch.Tensor(v)
            listener_sample_d[k] = v.unsqueeze(0)

        
        listener_sample_d["experiences"], target_decision_idx, orders = shuffle(listener_sample_d["experiences"])
        if not same:
            # The target_decision_idx is set to `nbr_experiences`:
            target_decision_idx = (dataset.nbr_distractors[dataset.mode]+1)*torch.ones(1).long()
        # shuffling the other keys similarly:
        for k,v in listener_sample_d.items():
            if k == "experiences":  continue
            listener_sample_d[k], _, _ = shuffle(v, orders=orders)
        
        ##--------------------------------------------------------------
        ##--------------------------------------------------------------

        output_dict = {"target_decision_idx":target_decision_idx}
        for k,v in listener_sample_d.items():
            output_dict[f"listener_{k}"] = v
        for k,v in speaker_sample_d.items():
            output_dict[f"speaker_{k}"] = v 

        return output_dict
