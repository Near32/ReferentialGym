from typing import Dict, List 

import os 

import torch
import torch.nn as nn
import torch.optim as optim 

from .module import Module
from ..networks import handle_nan, l1_reg, l2_reg

#TODO:
"""
1) Maybe make it possible for this module to ignore some task loss:
--> implement a mask-based policy?
"""

def build_OptimizationModule(id:str,
                             config:Dict[str,object],
                             input_stream_ids:Dict[str,str]=None) -> Module:
    return OptimizationModule(id=id,
                              config=config, 
                              input_stream_ids=input_stream_ids)


class OptimizationModule(Module):
    """
    Apply L1/L2 regularization by passing 'l1/2_reg_lambda' greater than 0.0.
    Only applied to parameters with non-null gradient, in order to leave parts of a 
    module that did not contribute to any loss's computation graph out.
    """

    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
                "losses_dict":"losses_dict",
                "logs_dict":"logs_dict",
                "mode":"signals:mode",
                "it_sample":"signals:it_sample",
                # step in the sequence of repetitions of the current batch
                "it_step":"signals:it_step",
                # step in the communication round.
            }

        assert "modules" in config,\
               "OptimizationModule relies on list of modules.\n\
                Not found in config."
        
        assert "optimizer_type" in config,\
               "OptimizationModule relies on 'optimizer_type'.\n\
                Not found in config."
        
        assert "mode" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'mode' id.\n\
                Not found in input_stream_ids."
        
        assert "losses_dict" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids."

        assert "logs_dict" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'logs_dict' id.\n\
                Not found in input_stream_ids."
        
        super(OptimizationModule, self).__init__(id=id,
                                                 type="OptimizationModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        self.update_count = 0
        parameters = []
        for k,m in self.config["modules"].items():
            parameters += m.parameters()
            #for name, param in m.named_parameters():
            #    parameters += param
            #    #print((name, param.shape))
            print(f"Module {k} of type {type(m)} : {len(list(m.parameters()))} params.")

        if "sgd" in self.config["optimizer_type"].lower():
            self.optimizer = optim.SGD(
                parameters, 
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            self.optimizer = optim.Adam(
                parameters, 
                lr=self.config["learning_rate"], 
                #betas=(0.9, 0.999), 
                weight_decay=self.config["weight_decay"],
                eps=self.config["adam_eps"],
            )

    def save(self, path):
      torch.save(self.optimizer.state_dict(), os.path.join(path, self.id+".module"))

    def load(self, path):
      self.optimizer.load_state_dict(torch.load(os.path.join(path, self.id+".module")))

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

        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]

        it_rep = input_streams_dict["it_sample"]
        it_comm_round = input_streams_dict["it_step"]

        for l_name, l in losses_dict.items():
            logs_dict[f"{mode}/{l_name}"] = l[-1]
        
        for l_name, l in losses_dict.items():
            losses_dict[l_name] = l[0]*l[-1]
        
        loss = sum([l.mean() for l in losses_dict.values()])

        if "train" in mode:
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config["l1_reg_lambda"] > 0.0:
                l1_regularization = {}
            if self.config["l2_reg_lambda"] > 0.0:
                l2_regularization = {}
            for k,m in self.config["modules"].items():
                m.apply(handle_nan)
                if self.config["with_gradient_clip"]:
                    nn.utils.clip_grad_value_(m.parameters(), self.config["gradient_clip"])
                if self.config["l1_reg_lambda"] > 0.0:
                    l1_reg(cum_loss_dict=l1_regularization, module=m)
                if self.config["l2_reg_lambda"] > 0.0:
                    l2_reg(cum_loss_dict=l2_regularization, module=m)
            if self.config["l1_reg_lambda"] > 0.0:
                l1_regularization = sum(l1_regularization.values())
                (l1_regularization*self.config["l1_reg_lambda"]).backward()
                logs_dict[f"{mode}/L1_regularization/loss"] = l1_regularization.item()
                logs_dict[f"{mode}/L1_regularization/lambda"] = self.config["l1_reg_lambda"]
            if self.config["l2_reg_lambda"] > 0.0:
                l2_regularization = sum(l2_regularization.values())
                (l2_regularization*self.config["l2_reg_lambda"]).backward()
                logs_dict[f"{mode}/L2_regularization/loss"] = l2_regularization.item()
                logs_dict[f"{mode}/L2_regularization/lambda"] = self.config["l2_reg_lambda"]
            
            self.optimizer.step()
            self.update_count += 1

        logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/Loss"] = loss
        
        outputs_stream_dict['signals:update_count'] = self.update_count
        
        return outputs_stream_dict
        
