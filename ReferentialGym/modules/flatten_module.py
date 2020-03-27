from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module


def build_FlattenModule(id:str,
                       input_stream_keys:List[str]) -> Module:
    return FlattenModule(id=id, 
                        input_stream_keys=input_stream_keys)


class FlattenModule(Module):
    def __init__(self,
                 id:str,
                 input_stream_keys:List[str]):

        input_stream_ids = {
                ik:f"input_{idx}" 
                for idx, ik in enumerate(input_stream_keys)
            }

        super(FlattenModule, self).__init__(id=f"FlattenModule_{id}",
                                            config=None,
                                            input_stream_keys=input_stream_keys,
                                            input_stream_ids=input_stream_ids)
        
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
            outputs_stream_dict[f'output_{idx}'] = inp.reshape(inp.shape[0], -1)
        
        return outputs_stream_dict
        