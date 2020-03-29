from typing import Dict, List 

import torch
import torch.nn as nn

from .module import Module


#TODO:
'''
1) Maybe make it possible for this module to ignore some task loss:
--> implement a mask-based policy?
'''

def build_HomoscedasticMultiTasksLossModule(id:str,
                                            config:Dict[str,object],
                                            input_stream_ids:Dict[str,str]=None) -> Module:
    return HomoscedasticMultiTasksLossModule(id=id,
                                             config=config, 
                                             input_stream_ids=input_stream_ids)


class HomoscedasticMultiTasksLossModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
            "losses_dict":"losses_dict"
            }

        assert("losses_dict" in input_stream_ids.values(), 
               "HomoscedasticMultiTasksLossModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids.")
        
        super(HomoscedasticMultiTasksLossModule, self).__init__(id=f"HomoscedasticMultiTasksLossModule_{id}",
                                                                config=config,
                                                                input_stream_ids=input_stream_ids)
        self.nbr_tasks = 2 #self.config['nbr_tasks']
        
        self.homoscedastic_log_vars = torch.nn.Parameter(torch.zeros(self.nbr_tasks))
        
        if self.config["use_cuda"]:
            self = self.cuda()

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

        loss_dict = input_streams_dict['losses_dict']

        nbr_tasks_ineffect = len(loss_dict)
        k0 = list(loss_dict.keys())[0]
        batch_size = loss_dict[k0][1].size()[0]

        if nbr_tasks_ineffect > self.nbr_tasks:
            self.nbr_tasks = nbr_tasks_ineffect
            self.homoscedastic_log_vars.data = torch.zeros(self.nbr_tasks).to(self.homoscedastic_log_vars.device)

        inv_uncertainty_sq = torch.exp( -self.homoscedastic_log_vars[:self.nbr_tasks])
        # (nbr_tasks)
        # (batch_size, self.nbr_tasks)
        batched_multiloss = {}
        for idx_loss, (kn, l) in enumerate(loss_dict.items()):
            batched_multiloss[kn] = inv_uncertainty_sq[idx_loss]*l[1]+self.homoscedastic_log_vars[idx_loss].repeat(l[1].shape[0])
        
        for kn in loss_dict:
            loss_dict[kn].append( batched_multiloss[kn])

        return outputs_stream_dict
        