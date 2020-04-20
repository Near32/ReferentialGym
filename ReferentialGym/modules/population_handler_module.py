from typing import Dict, List

import torch
import torch.nn as nn 

import os
import copy
import random 

from .module import Module


def build_PopulationHandlerModule(id:str,
                                  prototype_speaker:Module,
                                  prototype_listener:Module,
                                  config:Dict[str,object],
                                  input_stream_ids:Dict[str,str]) -> Module:
    
    return PopulationHandlerModule(id=id,
                                   prototype_speaker=prototype_speaker,
                                   prototype_listener=prototype_listener,
                                   config=config,
                                   input_stream_ids=input_stream_ids)


class PopulationHandlerModule(Module):
    def __init__(self,
                 id:str,
                 prototype_speaker:Module,
                 prototype_listener:Module,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]):

        assert "current_speaker_streams_dict" in input_stream_ids.values(),\
               "PopulationHandlerModule relies on 'current_speaker_streams_dict' id.\n\
                Not found in input_stream_ids."
        assert "current_listener_streams_dict" in input_stream_ids.values(),\
               "PopulationHandlerModule relies on 'current_listener_streams_dict' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "epoch" in input_stream_ids.values(),\
               "PopulationHandlerModule relies on 'epoch' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "global_it_datasample" in input_stream_ids.values(),\
               "PopulationHandlerModule relies on 'global_it_datasample' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "mode" in input_stream_ids.values(),\
               "PopulationHandlerModule relies on 'mode' id to compute its pipeline.\n\
                Not found in input_stream_ids."

        super(PopulationHandlerModule, self).__init__(id=id,
                                                    type="PopulationHandlerModule",
                                                    config=config,
                                                    input_stream_ids=input_stream_ids)
        
        print("Create Agents: ...")
        
        # Agents:
        nbr_speaker = self.config['cultural_speaker_substrate_size']
        self.speakers = nn.ModuleList()
        speakers = [prototype_speaker]+[ prototype_speaker.clone(clone_id=f's{i+1}') for i in range(nbr_speaker-1)]
        for speaker in speakers:
            self.speakers.append(speaker)
        nbr_listener = self.config['cultural_listener_substrate_size']
        self.listeners = nn.ModuleList()
        listeners = [prototype_listener]+[ prototype_listener.clone(clone_id=f'l{i+1}') for i in range(nbr_listener-1)]
        for listener in listeners:
            self.listeners.append(listener)

        if 'meta' in self.config['cultural_reset_strategy']:
            self.meta_agents = dict()
            self.meta_agents_optimizers = dict()
            for agent in [prototype_speaker, prototype_listener]: 
                if type(agent) not in self.meta_agents:
                    self.meta_agents[type(agent)] = agent.clone(clone_id=f'meta_{agent.agent_id}')
                    #self.meta_agents_optimizers[type(agent)] = optim.Adam(meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'], eps=self.config['adam_eps'])
                    self.meta_agents_optimizers[type(agent)] = optim.SGD(meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'])

        self.agents_stats = dict()
        for agent in self.speakers:
            self.agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        for agent in self.listeners:
            self.agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        
        print("Create Agents: OK.")

        self.previous_epoch = -1
        self.previous_global_it_datasample = -1
        self.counterGames = 0

    def _select_agents(self):
        idx_speaker = random.randint(0,len(self.speakers)-1)
        idx_listener = random.randint(0,len(self.listeners)-1)
            
        speaker = self.speakers[idx_speaker]
        listener = self.listeners[idx_listener]
        
        self.agents_stats[speaker.agent_id]['selection_iterations'].append(self.previous_global_it_datasample)
        self.agents_stats[listener.agent_id]['selection_iterations'].append(self.previous_global_it_datasample)

        return speaker, listener

    def bookkeeping(self, mode, epoch):
        it = self.previous_global_it_datasample
        
        if epoch != self.previous_epoch:
            self.previous_epoch = epoch
            # Save agent:
            for agent in self.speakers:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            for agent in self.listeners:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            
            if 'meta' in self.config['cultural_reset_strategy']:
                for agent in self.meta_agents.values():
                    agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))

        if 'train' in mode \
            and self.config["cultural_pressure_it_period"] is not None \
            and self.previous_global_it_datasample % self.config['cultural_pressure_it_period'] == 0:
            #and (idx_stimuli+len(data_loader)*epoch) % self.config['cultural_pressure_it_period'] == 0:
            if 'oldest' in self.config['cultural_reset_strategy']:
                if 'S' in self.config['cultural_reset_strategy']:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.speakers] 
                    idx_speaker2reset = random.choices( range(len(self.speakers)), weights=weights)[0]
                elif 'L' in self.config['cultural_reset_strategy']:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.listeners] 
                    idx_listener2reset = random.choices( range(len(self.listeners)), weights=weights)[0]
                else:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.listeners] 
                    weights += [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.speakers]
                    idx_agent2reset = random.choices( range(len(self.listeners)+len(self.speakers)), weights=weights)[0]
            else: #uniform
                if 'S' in self.config['cultural_reset_strategy']:
                    idx_speaker2reset = random.randint(0,len(self.speakers)-1)
                elif 'L' in self.config['cultural_reset_strategy']:
                    idx_listener2reset = random.randint(0,len(self.listeners)-1)
                else:
                    idx_agent2reset = random.randint(0,2*len(self.listeners)-1)

            if 'S' in self.config['cultural_reset_strategy']:
                if 'meta' in self.config['cultural_reset_strategy']:
                    self._apply_meta_update(meta_learner=self.meta_agents[type(self.speakers[idx_speaker2reset])],
                                           meta_optimizer=self.meta_agents_optimizers[type(self.speakers[idx_speaker2reset])],
                                           learner=self.speakers[idx_speaker2reset])
                else:
                    self.speakers[idx_speaker2reset].reset()
                self.agents_stats[self.speakers[idx_speaker2reset].agent_id]['reset_iterations'].append(it)
                print("Agent Speaker {} has just been resetted.".format(self.speakers[idx_speaker2reset].agent_id))
            
            if 'L' in self.config['cultural_reset_strategy']:
                if 'meta' in self.config['cultural_reset_strategy']:
                    self._apply_meta_update(meta_learner=self.meta_agents[type(self.listeners[idx_listener2reset])],
                                           meta_optimizer=self.meta_agents_optimizers[type(self.listeners[idx_listener2reset])],
                                           learner=self.listeners[idx_listener2reset])
                else:
                    self.listeners[idx_listener2reset].reset()
                self.agents_stats[self.listeners[idx_listener2reset].agent_id]['reset_iterations'].append(it)
                print("Agent  Listener {} has just been resetted.".format(self.listeners[idx_listener2reset].agent_id))

            if 'L' not in self.config['cultural_reset_strategy'] and 'S' not in self.config['cultural_reset_strategy']:
                if idx_agent2reset < len(self.listeners):
                    agents = self.listeners 
                else:
                    agents = self.speakers 
                    idx_agent2reset -= len(self.listeners)
                if 'meta' in self.config['cultural_reset_strategy']:
                    self._apply_meta_update(meta_learner=self.meta_agents[type(agents[idx_agent2reset])],
                                           meta_optimizer=self.meta_agents_optimizers[type(agents[idx_agent2reset])],
                                           learner=agents[idx_agent2reset])
                else:
                    agents[idx_agent2reset].reset()
                    self.agents_stats[agents[idx_agent2reset].agent_id]['reset_iterations'].append(it)
                print("Agents {} has just been resetted.".format(agents[idx_agent2reset].agent_id))
    
    def _reptile_step(self, learner, reptile_learner, nbr_grad_steps=1, verbose=False) :
        k = 1.0/float(nbr_grad_steps)
        nbrParams = 0
        nbrUpdatedParams = 0
        for (name, lp), (namer, lrp) in zip( learner.named_parameters(), reptile_learner.named_parameters() ) :
            nbrParams += 1
            if lrp.grad is not None:
                nbrUpdatedParams += 1
                lrp.grad.data.copy_( k*(lp.data-lrp.data) )
            else:
                lrp.grad = k*(lp.data-lrp.data)
                if verbose:
                    print("Parameter {} has not been updated...".format(name))
        print("Meta-cultural learning step :: {}/{} updated params.".format(nbrUpdatedParams, nbrParams))
        return 

    def _apply_meta_update(self, meta_learner, meta_optimizer, learner):
        meta_learner.zero_grad()
        self._reptile_step(learner=learner, reptile_learner=meta_learner)
        meta_optimizer.step()
        learner.load_state_dict( meta_learner.state_dict())
        return 


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

        assert input_streams_dict["current_speaker_streams_dict"]["ref"] is not None,\
                "It seems that you forgot to set a current_speaker using a CurrentAgentModule."
        assert input_streams_dict["current_listener_streams_dict"]["ref"] is not None,\
                "It seems that you forgot to set a current_listener using a CurrentAgentModule."

        outputs_stream_dict = {}
        
        epoch = input_streams_dict['epoch']
        mode = input_streams_dict['mode']
        global_it_datasample = input_streams_dict['global_it_datasample']
        
        if global_it_datasample != self.previous_global_it_datasample:
            self.bookkeeping(mode=mode, epoch=epoch)

            self.previous_global_it_datasample = global_it_datasample
            
            if 'train' in mode:
                self.counterGames += 1
                if 'obverter' in self.config['graphtype']:
                    # Let us decide whether to exchange the speakers and listeners:
                    # i.e. is the round of games finished?
                    if not('obverter_nbr_games_per_round' in self.config):
                        self.config['obverter_nbr_games_per_round'] = 1 
                    if  self.counterGames%self.config['obverter_nbr_games_per_round']==0:
                        # Invert the roles:
                        self.speakers, self.listeners = (self.listeners, self.speakers)
                        # Make it obvious to the stream handler:
                        outputs_stream_dict['speakers'] = self.speakers
                        outputs_stream_dict['listeners'] = self.listeners

                if self.config['iterated_learning_scheme'] and self.counterGames%self.config['iterated_learning_period']==0:
                    for lidx in range(len(self.listeners)):
                        self.listeners[lidx].reset()
                        print("Iterated Learning Scheme: Listener {} have just been resetted.".format(self.listeners[lidx].agent_id))
        
            new_speaker, new_listener = self._select_agents()
            
            if 'train' in mode: 
                new_speaker.train()
                new_listener.train()
            else:
                new_speaker.eval()
                new_listener.eval()

            new_speaker.role = "speaker"
            new_listener.role = "listener"

            input_streams_dict["current_speaker_streams_dict"]["ref"].set_ref(new_speaker)
            input_streams_dict["current_listener_streams_dict"]["ref"].set_ref(new_listener)

            if self.config['use_cuda']:
                input_streams_dict["current_speaker_streams_dict"]["ref"] = input_streams_dict["current_speaker_streams_dict"]["ref"].cuda()
                input_streams_dict["current_listener_streams_dict"]["ref"] = input_streams_dict["current_listener_streams_dict"]["ref"].cuda() 


        return outputs_stream_dict
