from typing import Dict, List, Optional

import torch
import torch.nn as nn 


class Module(nn.Module):
    def __init__(
        self,
        id:str,
        type:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str],
        output_stream_ids:Optional[Dict[str,str]]={},
    ):
        super(Module, self).__init__()
        self.id = id
        self.type = type
        self.config = config
        self.input_stream_ids = input_stream_ids
        self.output_stream_ids = output_stream_ids 

    def get_id(self) -> str:
        return self.id

    def get_type(self) -> str:
        return self.type 

    def get_input_stream_ids(self) -> Dict[str,str]:
        return self.input_stream_ids

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the requested 
        data stream, as values of the `self.input_stream_ids` dict.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: Dict[str, object] where the keys are the keys of 
                            `self.input_stream_ids`, and the values are the values
                            stored in the `StreamHandler`'s placeholder path defined
                            by the values of `self.input_stream_ids`.

        :returns:
            - output_streams_dict: 
        """
        raise NotImplementedError    

    def compute_pipeline_hooks(
        self, 
        input_streams_dict:Dict[str,object],
        outputs_dict:Dict[str,object],
    ) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the requested 
        data stream, as values of the `self.input_stream_ids` dict.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: Dict[str, object] where the keys are the keys of 
                            `self.input_stream_ids`, and the values are the values
                            stored in the `StreamHandler`'s placeholder path defined
                            by the values of `self.input_stream_ids`.

        :returns:
            - output_streams_dict: 
        """
        return None 

