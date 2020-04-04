from typing import Dict, List

import torch
import torch.nn as nn 

import copy

from .module import Module


def build_MultiHeadRegressionModule(id:str,
                                    config:Dict[str,object],
                                    input_stream_ids:Dict[str,str]) -> Module:
    
    # Multi Heads:
    # Add the feature map size as input to the architectures:
    feat_map_dim, feat_map_depth = config['input_stream_module']._compute_feature_shape()
    #flattened_feat_map_shape = 2 #feat_map_depth*feat_map_dim*feat_map_dim
    flattened_feat_map_shape = feat_map_depth*feat_map_dim*feat_map_dim
    for idx in range(len(config['heads_archs'])):
        config['heads_archs'][idx] = [flattened_feat_map_shape]+config['heads_archs'][idx]
    
    # Make sure there are as many heads as proposed architectures:
    while len(config['heads_archs']) != len(config['heads_output_sizes']):
        config['heads_archs'].append(copy.deepcopy(config['heads_archs'][-1]))

    # Add output sizes to the archs:
    for idx, output_size in enumerate(config['heads_output_sizes']):
        if isinstance(output_size, int):
            config['heads_archs'][idx].append(output_size)


    heads = nn.ModuleList()
    for idx, arch in enumerate(config['heads_archs']):
        if isinstance(config['heads_output_sizes'][idx], int):
            arch = config['heads_archs'][idx]
            sequence = []
            for i_l, (input_size, output_size) in enumerate(zip(arch,arch[1:])):
                sequence.append(nn.Linear(input_size, output_size))
                if i_l != len(arch)-2:
                    sequence.append(nn.ReLU())
            layer = nn.Sequential(*sequence)
        else:
            layer = None 
        heads.append(layer)
    
    module = MultiHeadRegressionModule(id=id,
                                       config=config,
                                       heads=heads,
                                       input_stream_ids=input_stream_ids)
    return module


class MultiHeadRegressionModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 heads:nn.ModuleList,
                 input_stream_ids:Dict[str,str],
                 final_fn:nn.Module=nn.Softmax(dim=-1)):

        assert("inputs" in input_stream_ids.values(), 
               "MultiHeadRegressionModule relies on 'inputs' id to start its pipeline.\n\
                Not found in input_stream_ids.")
        assert("targets" in input_stream_ids.values(), 
               "MultiHeadRegressionModule relies on 'targets' id to compute its pipeline.\n\
                Not found in input_stream_ids.")
        assert("losses_dict" in input_stream_ids.values(), 
               "MultiHeadRegressionModule relies on 'losses_dict' id to record the computated losses.\n\
                Not found in input_stream_ids.")
        assert("logs_dict" in input_stream_ids.values(), 
               "MultiHeadRegressionModule relies on 'logs_dict' id to record the accuracies.\n\
                Not found in input_stream_ids.")
        assert("loss_id" in config.keys(), 
               "MultiHeadRegressionModule relies on 'loss_id' key to record the computated losses and accuracies.\n\
                Not found in config keys.")

        super(MultiHeadRegressionModule, self).__init__(id=id,
                                                        type="MultiHeadRegressionModule",
                                                        config=config,
                                                        input_stream_ids=input_stream_ids)
        self.heads = heads
        
        if self.config["use_cuda"]:
            self = self.cuda()
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        '''
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        '''
        outputs_stream_dict = {}

        inputs = input_streams_dict['inputs']
        shape_inputs = inputs.shape
        batch_size = shape_inputs[0]
        
        flatten_input = inputs.view(batch_size, -1)
        if self.config['detach_feat_map']:
            flatten_input = flatten_input.detach()

        losses = []
        accuracies = []
        for ih, head in enumerate(self.heads):
            if isinstance(self.config['heads_output_sizes'][ih], int):
                head_output = head(flatten_input)

                # Loss:
                reg_target = inputs_dict['targets'][..., ih].float()
                reg_criterion = nn.SmoothL1Loss(reduction='none')
                reg_loss = reg_criterion( head_output, reg_target).squeeze()

                # Distance:
                distance = (head_output-reg_target).pow(2).sqrt().mean()
            else:
                loss = torch.zeros(batch_size).to(flatten_input.device)
                distance = torch.zeros(1)

            losses.append(loss)
            distances.append(distance)

        losses_dict = input_streams_dict['losses_dict']
        logs_dict = input_streams_dict['logs_dict'] 
        
        # MultiHead Reg Losses:
        for idx, loss in enumerate(losses):
            losses_dict[f"{self.config['loss_id']}/multi_reg_head_{idx}_loss"] = [1e3, loss]

        # MultiHead Reg Distance:
        for idx, dist in enumerate(reg_distances):
            log_dict[f"{self.config['loss_id']}/multi_reg_head_{idx}_distance"] = dist

        outputs_stream_dict['losses'] = losses
        outputs_stream_dict['distances'] = distances

        return outputs_stream_dict
