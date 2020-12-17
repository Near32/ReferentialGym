from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module
from ..networks import handle_nan

#TODO:
"""
1) Maybe make it possible for this module to ignore some task loss:
--> implement a mask-based policy?
"""

def build_OneHotEncoderModule(id:str,
                             config:Dict[str,object],
                             input_stream_ids:Dict[str,str]) -> Module:
    return OneHotEncoderModule(id=id,
                              config=config, 
                              input_stream_ids=input_stream_ids)


class OneHotEncoderModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]):
        '''
        One-hot encodes input streams data.

        :param config:
          - `'nbr_values'`: integer specifying the number of values possible in the one-hot encoding.
          - `'flatten'`: bool specifying whether to flatten the representation or leaving it as is.
        '''
        super(OneHotEncoderModule, self).__init__(id=id,
                                                 type="OneHotEncoderModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
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

        for idx, (k, inp) in enumerate(input_streams_dict.items()):
            batch_size = inp.shape[0]
            ohe_inp = torch.eye(self.config['nbr_values'])[inp]
            
            if self.config['flatten']:
              ohe_inp = ohe_inp.reshape((batch_size, -1))

            outputs_stream_dict[k] = ohe_inp

        return outputs_stream_dict
        