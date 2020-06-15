from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module


def build_ConcatModule(id:str,
                       config:Dict[str,object],
                       input_stream_keys:List[str]) -> Module:
    return ConcatModule(id=id,
                        config=config, 
                        input_stream_keys=input_stream_keys)


class ConcatModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_keys:List[str]):

        input_stream_ids = {
                ik:f"input_{idx}" 
                for idx, ik in enumerate(input_stream_keys)
            }

        assert("dim" in config, 
               "ConcatModule relies on 'dim' value.\n\
                Not found in config.")
        
        super(ConcatModule, self).__init__(id=id,
                                           type="ConcatModule",
                                           config=config,
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

        outputs_stream_dict["output_0"] = torch.cat(list(input_streams_dict.values()), dim=self.config["dim"])
        
        return outputs_stream_dict
        