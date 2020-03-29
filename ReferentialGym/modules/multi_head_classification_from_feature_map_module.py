from typing import Dict, List

import torch
import torch.nn as nn 

import copy

from .module import Module


def build_MultiHeadClassificationFromFeatureMapModule(id:str,
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
    
    module = MultiHeadClassificationFromFeatureMapModule(id=id,
                                           config=config,
                                           heads=heads,
                                           input_stream_ids=input_stream_ids)
    return module


class MultiHeadClassificationFromFeatureMapModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 heads:nn.ModuleList,
                 input_stream_ids:Dict[str,str],
                 final_fn:nn.Module=nn.Softmax(dim=-1)):

        assert("inputs" in input_stream_ids.values(), 
               "ClassificationModule relies on 'inputs' id to start its pipeline.\n\
                Not found in input_stream_ids.")
        assert("targets" in input_stream_ids.values(), 
               "ClassificationModule relies on 'targets' id to compute its pipeline.\n\
                Not found in input_stream_ids.")
        assert("losses_dict" in input_stream_ids.values(), 
               "ClassificationModule relies on 'losses_dict' id to record the computated losses.\n\
                Not found in input_stream_ids.")
        assert("logs_dict" in input_stream_ids.values(), 
               "ClassificationModule relies on 'logs_dict' id to record the accuracies.\n\
                Not found in input_stream_ids.")
        assert("loss_id" in config.keys(), 
               "ClassificationModule relies on 'loss_id' key to record the computated losses and accuracies.\n\
                Not found in config keys.")

        super(MultiHeadClassificationFromFeatureMapModule, self).__init__(id=f"MultiHeadClassificationFromFeatureMapModule_{id}",
                                                            config=config,
                                                            input_stream_ids=input_stream_ids)
        self.heads = heads
        self.final_fn = final_fn

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
                final_output = self.final_fn(head_output)

                # Loss:
                target_idx = input_streams_dict['targets'][..., ih].squeeze()
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion( final_output, target_idx)

                # Accuracy:
                argmax_final_output = final_output.argmax(dim=-1)
                accuracy = 100.0*(target_idx==argmax_final_output).float().mean()
            else:
                loss = torch.zeros(batch_size).to(flatten_input.device)
                accuracy = torch.zeros(1)

            losses.append(loss)
            accuracies.append(accuracy)

        losses_dict = input_streams_dict['losses_dict']
        logs_dict = input_streams_dict['logs_dict'] 
        
        # MultiHead Losses:
        for idx, loss in enumerate(losses):
            losses_dict[f"{self.config['loss_id']}/multi_head_{idx}_loss"] = [1.0, loss]

        # MultiHead Accuracy:
        for idx, acc in enumerate(accuracies):
            logs_dict[f"{self.config['loss_id']}/multi_head_{idx}_accuracy"] = acc

        outputs_stream_dict['losses'] = losses
        outputs_stream_dict['accuracies'] = accuracies

        return outputs_stream_dict
