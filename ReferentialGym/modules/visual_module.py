from typing import Dict, List 

import torch
import torch.nn as nn

from .module import Module
from ..networks import choose_architecture, BetaVAE

def build_VisualModule(id:str,
                       config:Dict[str,object],
                       input_stream_ids:Dict[str,str]) -> Module:
    
    obs_shape = config["obs_shape"]

    cnn_input_shape = obs_shape[2:]
    MHDPANbrHead=4
    MHDPANbrRecUpdate=1
    MHDPANbrMLPUnit=512
    MHDPAInteractionDim=128
    if "mhdpa_nbr_head" in config: MHDPANbrHead = config["mhdpa_nbr_head"]
    if "mhdpa_nbr_rec_update" in config: MHDPANbrRecUpdate = config["mhdpa_nbr_rec_update"]
    if "mhdpa_nbr_mlp_unit" in config: MHDPANbrMLPUnit = config["mhdpa_nbr_mlp_unit"]
    if "mhdpa_interaction_dim" in config: MHDPAInteractionDim = config["mhdpa_interaction_dim"]
    
    encoder = choose_architecture(
        architecture=config["architecture"],
        kwargs=config,
        input_shape=cnn_input_shape,
        feature_dim=config["cnn_encoder_feature_dim"],
        nbr_channels_list=config["cnn_encoder_channels"],
        kernels=config["cnn_encoder_kernels"],
        strides=config["cnn_encoder_strides"],
        paddings=config["cnn_encoder_paddings"],
        fc_hidden_units_list=config["cnn_encoder_fc_hidden_units"],
        dropout=config["dropout_prob"],
        MHDPANbrHead=MHDPANbrHead,
        MHDPANbrRecUpdate=MHDPANbrRecUpdate,
        MHDPANbrMLPUnit=MHDPANbrMLPUnit,
        MHDPAInteractionDim=MHDPAInteractionDim
    )

    use_feat_converter = config["use_feat_converter"] if "use_feat_converter" in config else False 
    if use_feat_converter:
        featout_converter_input = encoder.get_feature_shape()

    encoder_feature_shape = encoder.get_feature_shape()
    if use_feat_converter:
        featout_converter = []
        featout_converter.append(nn.Linear(featout_converter_input, config["cnn_encoder_feature_dim"]*2))
        featout_converter.append(nn.ReLU())
        featout_converter.append(nn.Linear(config["cnn_encoder_feature_dim"]*2, config["feat_converter_output_size"])) 
        featout_converter.append(nn.ReLU())
        featout_converter =  nn.Sequential(*featout_converter)
        encoder_feature_shape = config["feat_converter_output_size"]
    else:
        featout_converter = None 

    featout_normalization = nn.BatchNorm1d(num_features=encoder_feature_shape)

    module = VisualModule(id=id,
                          encoder=encoder,
                          featout_converter=featout_converter,
                          featout_normalization=featout_normalization,
                          config=config,
                          input_stream_ids=input_stream_ids)
    print(module)
    
    return module

class VisualModule(Module):
    def __init__(self, 
                 id, 
                 encoder,
                 featout_converter,
                 featout_normalization,
                 config,
                 input_stream_ids,
                 ):
        
        assert "inputs" in input_stream_ids.keys(),\
               "VisualModule relies on 'inputs' id to start its pipeline.\n\
                Not found in input_stream_ids."
        assert "losses_dict" in input_stream_ids.keys(),\
               "VisualModule relies on 'losses_dict' id to record the computated losses.\n\
                Not found in input_stream_ids."
        assert "logs_dict" in input_stream_ids.keys(),\
               "VisualModule relies on 'logs_dict' id to record the accuracies.\n\
                Not found in input_stream_ids."
        assert "mode" in input_stream_ids.keys(),\
               "VisualModule relies on 'mode' key to record the computated losses and accuracies.\n\
                Not found in input_stream_ids."

        
        super(VisualModule, self).__init__(id=id, 
            type="VisualModule", 
            config=config, 
            input_stream_ids=input_stream_ids)
        
        self.encoder = encoder 
        self.featout_converter = featout_converter
        self.featout_normalization = featout_normalization

        if "BetaVAE" in self.config["architecture"] or "MONet" in self.config["architecture"]:
            self.VAE_losses = list()
            self.compactness_losses = list()
            self.buffer_cnn_output_dict = dict()
            
            self.VAE = self.encoder
        
        if self.config["use_cuda"]:
            self = self.cuda()
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        mode = input_streams_dict["mode"]
        self.experiences = experiences = input_streams_dict["inputs"]
        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]

        batch_size = experiences.size(0)
        nbr_distractors_po = experiences.size(1)
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
        feat_maps = []
        total_size = experiences.size(0)
        mini_batch_size = min(self.config["cnn_encoder_mini_batch_size"], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            if isinstance(self.encoder, BetaVAE):
                cnn_output_dict  = self.encoder.compute_loss(stin)
                if "VAE_loss" in cnn_output_dict:
                    self.VAE_losses.append(cnn_output_dict["VAE_loss"])
                
                if hasattr(self.encoder, "compactness_losses") and self.encoder.compactness_losses is not None:
                    self.compactness_losses.append(self.encoder.compactness_losses.cpu())
                
                for key in cnn_output_dict:
                    if key not in self.buffer_cnn_output_dict:
                        self.buffer_cnn_output_dict[key] = list()
                    self.buffer_cnn_output_dict[key].append(cnn_output_dict[key].cpu())

                if self.config["vae_use_mu_value"]:
                    featout = self.encoder.mu 
                else:
                    featout = self.encoder.z

                feat_map = self.encoder.get_feat_map()
            else:
                featout = self.encoder(stin)
                feat_map = self.encoder.get_feat_map()
            
            if self.featout_converter is not None:
                featout = self.featout_converter(featout)

            features.append(featout)
            feat_maps.append(feat_map)

        self.features_not_normalized = torch.cat(features, dim=0)
        self.features = self.featout_normalization(self.features_not_normalized)
        self.feat_maps = torch.cat(feat_maps, dim=0)
        
        self.features = self.features.view(batch_size, nbr_distractors_po, self.config["nbr_stimulus"], -1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        outputs_stream_dict["feat_maps"] = self.feat_maps 
        outputs_stream_dict["features"] = self.features
        
        if isinstance(self.encoder, BetaVAE):
            VAE_losses = torch.cat(self.VAE_losses).contiguous()
            losses_dict[f"{mode}/{self.id}/VAE_loss"] = [self.config["VAE_lambda"], VAE_losses]

            for key in self.buffer_cnn_output_dict:
                logs_dict[f"{mode}/{self.id}/{key}"] = torch.cat(self.buffer_cnn_output_dict[key]).mean()

            logs_dict[f"{mode}/{self.id}/kl_capacity"] = torch.Tensor(
                [100.0*self.encoder.EncodingCapacity/self.encoder.maxEncodingCapacity]
            )
            if len(self.compactness_losses):
                logs_dict[f"{mode}/{self.id}/unsup_compactness_loss"] = torch.cat(self.compactness_losses).mean()
            
            # resetting:
            self.VAE_losses = list()
            self.compactness_losses = list()
            self.buffer_cnn_output_dict = dict()
            
        return outputs_stream_dict 