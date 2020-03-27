from typing import Dict, List

import torch
import torch.nn as nn 


class Module(nn.Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_keys:List[str],
                 input_stream_ids:Dict[str,str]):
        super(Module, self).__init__()
        self.id = id
        self.config = config
        self.input_stream_keys = input_stream_keys 
        self.input_stream_ids = input_stream_ids

    def get_id(self) -> str:
        return self.id 

    def get_input_stream_keys(self) -> List[str]:
        return self.input_stream_keys

    def get_input_stream_ids(self) -> Dict[str,str]:
        return self.input_stream_ids

    def compute(self, inputs_dict:Dict[str,object]) -> Dict[str,object] :
        '''
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param inputs_dict: dict of str and data elements that 
                            follows `self.input_stream_ids`'s keywords
                            and are extracted from `self.input_stream_keys`
                            -named streams.

        :returns:
            - outputs_sream_dict: 
        '''
        raise NotImplementedError