from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module


def build_BatchReshapeRepeatModule(id:str,
                             config:Dict[str,object],
                             input_stream_keys:List[str]) -> Module:
    return BatchReshapeRepeatModule(id=id,
                              config=config, 
                              input_stream_keys=input_stream_keys)


class BatchReshapeRepeatModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_keys:List[str]):
        """
        Reshape input streams data while keeping the batch dimension identical.

        :param config: Dict of parameters. Expectes:
            - "new_shape": List of None/Tuple/List/torch.Size representing the new shape
                of each input stream, without mentionning the batch dimension.
                If multiple input streams are proposed but only one element in this
                list, then the list is expanded by repeating the last element.
            - "repetition": List of None/Tuple/List/torch.Size representing the repetition
                of each input stream, without mentionning the batch dimension.
                If multiple input streams are proposed but only one element in this
                list, then the list is expanded by repeating the last element.
        """
        input_stream_ids = {
                f"input_{idx}":ik
                for idx, ik in enumerate(input_stream_keys)
            }

        assert "new_shape" in config,\
               "BatchReshapeRepeatModule relies on 'new_shape' list.\n\
                Not found in config."

        assert "repetition" in config,\
               "BatchReshapeRepeatModule relies on 'repetition' list.\n\
                Not found in config."

        super(BatchReshapeRepeatModule, self).__init__(id=id,
                                                       type="BatchReshapeRepeatModule",
                                                       config=config,
                                                       input_stream_ids=input_stream_ids)
                
        self.new_shape = self.config["new_shape"]
        assert isinstance(self.new_shape, list)
        self.repetition = self.config["repetition"]
        assert isinstance(self.repetition, list)
        
        self.n_input_streams = len(self.input_stream_ids)

        while len(self.new_shape) < self.n_input_streams:
            self.new_shape.append(self.new_shape[-1])

        while len(self.repetition) < self.n_input_streams:
            self.repetition.append(self.repetition[-1])

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`"s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        for idx, (k, inp) in enumerate(input_streams_dict.items()):
            new_shape = self.new_shape[idx]
            if new_shape is None:
                new_shape = inp.shape[1:] 
            n_inp = inp.reshape(inp.shape[0], *new_shape)

            repeat = self.repetition[idx]
            if repeat is not None:
                n_inp = n_inp.repeat(1, *repeat) 
            
            outputs_stream_dict[f"output_{idx}"] = n_inp
        
        return outputs_stream_dict
        