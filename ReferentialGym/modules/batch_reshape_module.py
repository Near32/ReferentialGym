from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module


def build_BatchReshapeModule(id:str,
                             config:Dict[str,object],
                             input_stream_keys:List[str]) -> Module:
    return BatchReshapeModule(id=id,
                              config=config, 
                              input_stream_keys=input_stream_keys)


class BatchReshapeModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_keys:List[str]):
        '''
        Reshape input streams data while keeping the batch dimension identical.

        :param config: Dict of parameters. Expectes:
            - "new_shape": List of Tuple/List/torch.Size representing the new shape
                of each input stream, without mentionning the batch dimension.
                If multiple input streams are proposed but only one element in this
                list, then the list is expanded by repeatition.
        '''
        input_stream_ids = {
                ik:f"input_{idx}" 
                for idx, ik in enumerate(input_stream_keys)
            }

        assert("new_shape" in config, 
               "BatchReshapeModule relies on 'new_shape' list.\n\
                Not found in config.")

        super(BatchReshapeModule, self).__init__(id=f"BatchReshapeModule_{id}",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.new_shape = self.config["new_shape"]
        assert(isinstance(self.new_shape, list))
        self.n_input_streams = len(self.input_stream_keys)

        while len(self.new_shape) < self.n_input_streams:
            self.new_shape.append(self.new_shape[-1])

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
            outputs_stream_dict[f'output_{idx}'] = inp.reshape(inp.shape[0], *self.new_shape[idx])
        
        return outputs_stream_dict
        