from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module


def build_SqueezeModule(id:str,
                       config:Dict[str,object],
                       input_stream_keys:List[str]) -> Module:
    return SqueezeModule(id=id,
                        config=config, 
                        input_stream_keys=input_stream_keys)


class SqueezeModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_keys:List[str]):
        '''
        Squeeze input streams data (beware the batch dimension if it is equal to 1...).

        :param config: Dict of parameters. Expectes:
            - "dim": List of None/Tuple/List/torch.Size representing the index
                of the dimension to squeeze for each input stream.
                If multiple input streams are proposed but only one element in this
                list, then the list is expanded by repeating the last element.
        '''
        
        input_stream_ids = {
                ik:f"input_{idx}" 
                for idx, ik in enumerate(input_stream_keys)
            }

        assert("dim" in config, 
               "SqueezeModule relies on 'dim' value.\n\
                Not found in config.")
        
        super(SqueezeModule, self).__init__(id=f"SqueezeModule_{id}",
                                           config=config,
                                           input_stream_ids=input_stream_ids)
        
        self.squeeze_dim = self.config["dim"]
        assert(isinstance(self.squeeze_dim, list))
        
        self.n_input_streams = len(self.input_stream_ids)

        while len(self.squeeze_dim) < self.n_input_streams:
            self.squeeze_dim.append(self.squeeze_dim[-1])

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

        for idx, (k, inp) in enumerate(input_streams_dict.items()):
            if self.squeeze_dim[idx] is not None:
                n_inp = inp.squeeze(dim=self.squeeze_dim[idx])
            else:
                n_inp = inp.squeeze()

            outputs_stream_dict[f'output_{idx}'] = n_inp

        return outputs_stream_dict
        