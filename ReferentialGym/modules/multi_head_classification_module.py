from typing import Dict, List

import torch
import torch.nn as nn 

import copy

from .module import Module


def build_MultiHeadClassificationModule(id:str,
                                        config:Dict[str,object],
                                        input_stream_ids:Dict[str,str]) -> Module:
    
    # Multi Heads:
    # Add the feature map size as input to the architectures:
    input_shapes = config["input_shapes"]
    for idx in range(len(config["heads_archs"])):
        config["heads_archs"][idx] = [input_shapes[idx]]+config["heads_archs"][idx]
        if config["same_head"]:
            break

    # Make sure there are as many heads as proposed architectures:
    while len(config["heads_archs"]) != len(config["heads_output_sizes"]):
        config["heads_archs"].append(copy.deepcopy(config["heads_archs"][-1]))

    # Add output sizes to the archs:
    for idx, output_size in enumerate(config["heads_output_sizes"]):
        config["heads_archs"][idx].append(output_size)
        if config["same_head"]:
            break


    heads = nn.ModuleList()
    if config['same_head']:
        #config['heads_output_sizes'][0] = max(config['heads_output_sizes'])
        config['heads_archs'][0][-1] = max(config['heads_output_sizes'])
    
    for idx, arch in enumerate(config["heads_archs"]):
        if isinstance(config["heads_output_sizes"][idx], int):
            arch = config["heads_archs"][idx]
            sequence = []
            input_size = arch[0]
            for i_l, output_size in enumerate(arch[1:]):
                add_DP = False
                DP_thresh = 0.0
                if isinstance(output_size,str):
                    if "DP" in output_size:
                        # Assumes "X-DPY" 
                        # where Y is DP_thresh and X is nbr neurons...
                        add_DP = True
                        output_size = output_size.split("-")
                        DP_thresh = float(output_size[1].replace("DP",""))
                        output_size = int(output_size[0])
                    else:
                        raise NotImplementedError
                sequence.append(nn.Linear(input_size, output_size))
                if i_l != len(arch)-2:
                    sequence.append(nn.ReLU())
                if add_DP:
                    sequence.append(nn.Dropout(p=DP_thresh))
                input_size = output_size

            layer = nn.Sequential(*sequence)
        else:
            layer = None 
        heads.append(layer)
        
        if config["same_head"]:
            break
    
    module = MultiHeadClassificationModule(id=id,
                                           config=config,
                                           heads=heads,
                                           input_stream_ids=input_stream_ids)
    print(module)

    return module


class MultiHeadClassificationModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 heads:nn.ModuleList,   
                 input_stream_ids:Dict[str,str],
                 final_fn:nn.Module=nn.Softmax(dim=-1)):

        assert "inputs_0" in input_stream_ids.keys(),\
               "ClassificationModule relies on, at least, 'input_0' id to start its pipeline.\n\
                Not found in input_stream_ids."
        assert "targets_0" in input_stream_ids.keys(),\
               "ClassificationModule relies on, at least, 'targets_0' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "losses_dict" in input_stream_ids.keys(),\
               "ClassificationModule relies on 'losses_dict' id to record the computated losses.\n\
                Not found in input_stream_ids."
        assert "logs_dict" in input_stream_ids.keys(),\
               "ClassificationModule relies on 'logs_dict' id to record the accuracies.\n\
                Not found in input_stream_ids."
        assert "mode" in input_stream_ids.keys(),\
               "ClassificationModule relies on 'mode' key to record the computed losses and accuracies.\n\
                Not found in input_stream_ids."
        assert "loss_ids" in config.keys(),\
               "ClassificationModule relies on 'loss_ids' key to record the computed losses and accuracies.\n\
                Not found in config keys."
        
        super(MultiHeadClassificationModule, self).__init__(id=id, 
            type="MultiHeadClassificationModule", 
            config=config, 
            input_stream_ids=input_stream_ids)
        
        self.heads = heads
        self.final_fn = final_fn

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

        inputs = {k:v for k,v in input_streams_dict.items() if "inputs" in k}
        nbr_inputs = len(inputs)
        
        shape_inputs = {k:v.shape for k,v in inputs.items()}
        batch_sizes = {k:sh[0] for k,sh in shape_inputs.items()}
        
        need_reshaping = {k:False for k in inputs}
        for k, input_shape in shape_inputs.items():
            if len(input_shape) > 2:
                """
                Let us reshape the input so that it only has 2 dimensions.
                Afterwards, the resulting outputs will be reshaped and 
                averaged over the extra dimensions.
                N.B: the target is reshaped similarly...
                """
                need_reshaping[k] = True
                last_dim_size = input_shape[-1]
                inputs[k] = inputs[k].reshape(-1, last_dim_size)

        losses = {}
        accuracies = {}
        predicted_labels = {}
        groundtruth_labels = {}

        for ii, (key,inp) in enumerate(inputs.items()):
            if self.config["same_head"]:    ih = 0
            else:   ih = ii
            head = self.heads[ih]

            head_output = head(inp)
            final_output = self.final_fn(head_output)

            # Loss:
            target_idx = input_streams_dict[f"targets_{ii}"]
            if need_reshaping[key]:
                # Target indices corresponds to 1 per batch element:
                target_idx = target_idx.reshape(-1)          
            
            criterion = nn.CrossEntropyLoss(reduction="none")
            loss = criterion( final_output, target_idx.squeeze().long())

            # Accuracy:
            argmax_final_output = final_output.argmax(dim=-1)
            accuracy = 100.0*(target_idx==argmax_final_output).float()

            if need_reshaping[key]:
                loss = loss.reshape(batch_sizes[key],-1).mean(-1)
        
            losses[key] = loss
            accuracies[key] = accuracy

            predicted_labels[self.config['loss_ids'][key]] = argmax_final_output.cpu().detach().numpy()
            groundtruth_labels[self.config['loss_ids'][key]] = target_idx.cpu().detach().numpy()

        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"] 
        
        # MultiHead Losses:
        for key, loss in losses.items():
            losses_dict[f"{self.config['loss_ids'][key]}/loss"] = [self.config.get('loss_lambdas',{}).get(key, 1.0), loss]

        # MultiHead Accuracy:
        for key, acc in accuracies.items():
            logs_dict[f"{mode}/{self.config['loss_ids'][key]}/accuracy"] = acc.mean()

        for group_key, keys in self.config['grouped_accuracies'].items():
            acc = torch.stack([accuracies[key_acc] for key_acc in keys]).mean()
            logs_dict[f"{mode}/{group_key}/accuracy"] = acc.mean()


        outputs_stream_dict["losses"] = losses
        outputs_stream_dict["accuracies"] = accuracies

        outputs_stream_dict["predicted_labels"] = predicted_labels
        outputs_stream_dict["groundtruth_labels"] = groundtruth_labels

        return outputs_stream_dict
