from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module
from ..networks import handle_nan

#TODO:
'''
1) Maybe make it possible for this module to ignore some task loss:
--> implement a mask-based policy?
'''

def build_OptimizationModule(id:str,
                             config:Dict[str,object],
                             input_stream_ids:Dict[str,str]=None) -> Module:
    return OptimizationModule(id=id,
                              config=config, 
                              input_stream_ids=input_stream_ids)


class OptimizationModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
                "losses_dict":"losses_dict",
                "signal:mode":"mode",
            }

        assert("modules" in config, 
               "OptimizationModule relies on list of modules.\n\
                Not found in config.")
        
        assert("mode" in input_stream_ids.values(), 
               "OptimizationModule relies on 'mode' id.\n\
                Not found in input_stream_ids.")
        
        assert("losses_dict" in input_stream_ids.values(), 
               "OptimizationModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids.")
        
        super(OptimizationModule, self).__init__(id=f"OptimizationModule_{id}",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        parameters = []
        for k,m in self.config["modules"].items():
            parameters += m.parameters()

        self.optimizer = optim.Adam(parameters, 
                                    lr=self.config['learning_rate'], 
                                    betas=(0.9, 0.999), 
                                    eps=self.config['adam_eps'])
    
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

        losses_dict = input_streams_dict['losses_dict']
        mode = input_streams_dict['mode']

        for k, v in losses_dict.items():
            losses_dict[k][-1] = v[0]*v[-1].mean()
        
        loss = sum([l[-1] for l in losses_dict.values()])

        if 'train' in mode:
            loss.backward()
            
            for k,m in self.config["modules"].items():
                m.apply(handle_nan)
                if self.config['with_gradient_clip']:
                    nn.utils.clip_grad_value_(m.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        
        return outputs_stream_dict
        