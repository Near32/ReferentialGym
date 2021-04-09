from typing import Dict, List

import torch
import torch.nn as nn 
import torch.optim as optim 

import os
import pickle
import copy
import random 
import glob

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

        assert "current_speaker_streams_dict" in input_stream_ids.keys(),\
               "PopulationHandlerModule relies on 'current_speaker_streams_dict' id.\n\
                Not found in input_stream_ids."
        assert "current_listener_streams_dict" in input_stream_ids.keys(),\
               "PopulationHandlerModule relies on 'current_listener_streams_dict' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "epoch" in input_stream_ids.keys(),\
               "PopulationHandlerModule relies on 'epoch' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "global_it_datasample" in input_stream_ids.keys(),\
               "PopulationHandlerModule relies on 'global_it_datasample' id to compute its pipeline.\n\
                Not found in input_stream_ids."
        assert "mode" in input_stream_ids.keys(),\
               "PopulationHandlerModule relies on 'mode' id to compute its pipeline.\n\
                Not found in input_stream_ids."

        super(PopulationHandlerModule, self).__init__(id=id,
                                                    type="PopulationHandlerModule",
                                                    config=config,
                                                    input_stream_ids=input_stream_ids)
        
        print("Create Agents: ...")

        self.verbose = config["verbose"]
        
        # Agents:
        if 'cultural_speaker_substrate_size' not in self.config:
            self.config['cultural_speaker_substrate_size'] = 1
        nbr_speaker = self.config['cultural_speaker_substrate_size']
        self.speakers = nn.ModuleList()
        self.dspeakers = dict()
        speakers = [prototype_speaker]+[ prototype_speaker.clone(clone_id=f's{i+1}') for i in range(nbr_speaker-1)]
        for speaker in speakers:
            self.speakers.append(speaker)
            self.dspeakers[speaker.id] = speaker
        if 'cultural_listener_substrate_size' not in self.config:
            self.config['cultural_listener_substrate_size'] = 1
        nbr_listener = self.config['cultural_listener_substrate_size']
        self.listeners = nn.ModuleList()
        self.dlisteners = dict()
        listeners = [prototype_listener]+[ prototype_listener.clone(clone_id=f'l{i+1}') for i in range(nbr_listener-1)]
        for listener in listeners:
            self.listeners.append(listener)
            self.dlisteners[listener.id] = listener

        if 'cultural_reset_strategy' in self.config\
            and 'meta' in self.config['cultural_reset_strategy']:
            self.meta_agents = dict()
            self.meta_agents_optimizers = dict()
            for agent in [prototype_speaker, prototype_listener]: 
                if type(agent) not in self.meta_agents:
                    self.meta_agents[type(agent)] = agent.clone(clone_id=f'meta_{agent.agent_id}')
                    #self.meta_agents_optimizers[type(agent)] = optim.Adam(self.meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'], eps=self.config['adam_eps'])
                    self.meta_agents_optimizers[type(agent)] = optim.SGD(self.meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'])

        self.agents_stats = dict()
        for agent in self.speakers:
            self.agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        for agent in self.listeners:
            self.agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        
        print("Create Agents: OK.")

        self.previous_epoch = -1
        self.previous_global_it_datasample = {}
        self.counterGames = 0

    def save(self, path):
        path = os.path.join(path, self.id)
        os.makedirs(path, exist_ok=True)

        if 'cultural_reset_strategy' in self.config\
            and 'meta' in self.config['cultural_reset_strategy']:
            meta_agents_optimizers = self.meta_agents_optimizers
            self.meta_agents_optimizers = None
            meta_agents_optimizers_state_dicts = {k: v.state_dict() for k, v in meta_agents_optimizers.items()}
            torch.save(meta_agents_optimizers_state_dicts, path+".optimizers_state_dict")
            self.meta_agents_optimizers = meta_agents_optimizers
            
            meta_path = os.path.join(path, "meta_agents")
            os.makedirs(meta_path, exist_ok=True)

            for name, meta_agent in self.meta_agents:
                meta_agent.save(meta_path, filename=meta_agent.id+".agent")
        
        speakers_path = os.path.join(path, "speakers")
        os.makedirs(speakers_path, exist_ok=True)
        for speaker in self.speakers:
            speaker.save(speakers_path, filename=speaker.id+".agent")
        
        listeners_path = os.path.join(path, "listeners")
        os.makedirs(listeners_path, exist_ok=True)
        for listener in self.listeners:
            listener.save(listeners_path, filename=listener.id+".agent")

        try:
            with open(path+"agent_stats.dict", 'wb') as f:
                pickle.dump(self.agents_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save agents stats: {e}")

            
    def load(self, path):
        mpath = os.path.join(path, self.id)

        try:
            with open(mpath+"agent_stats.dict", 'rb') as f:
                self.agents_stats = pickle.load(f)
        except Exception as e:
            print(f"Exception caught while trying to load agents stats: {e}")

        listeners_path = os.path.join(mpath, "listeners")
        listeners_paths = glob.glob(os.path.join(listeners_path, "*.agent"))
        for listener_idx, listener_path in enumerate(listener_paths):
            try:
                listener = torch.load(listener_path)
                listener_id = listener.id 
                if listener_id not in self.dlisteners.keys():
                    print(f"WARNING: loading a listener that was not there previously...: {listener_id}.")
                    raise
                self.dlisteners[meta_agent_id] = listener
                self.listeners[listener_idx] = listener
            except Exception as e:
                print(f"WARNING: exception caught when trying to load listener {listener_id}: {e}")

        speakers_path = os.path.join(mpath, "speakers")
        speakers_paths = glob.glob(os.path.join(speakers_path, "*.agent"))
        for speaker_idx, speaker_path in enumerate(speaker_paths):
            try:
                speaker = torch.load(speaker_path)
                speaker_id = speaker.id 
                if speaker_id not in self.dspeakers.keys():
                    print(f"WARNING: loading a speaker that was not there previously...: {speaker_id}.")
                    raise
                self.dspeakers[meta_agent_id] = speaker
                self.speakers[speaker_idx] = speaker
            except Exception as e:
                print(f"WARNING: exception caught when trying to load speaker {speaker_id}: {e}")

        if 'cultural_reset_strategy' in self.config\
            and 'meta' in self.config['cultural_reset_strategy']:
            meta_agents_optimizers_state_dicts = torch.load(mpath+".optimizers_state_dict")
            for k,v in meta_agents_optimizers_state_dicts.items():
                try:
                    self.meta_agents_optimizers[k].load_state_dict(v)
                except Exception as e:
                    print(f"WARNING: exception caught when trying to load meta agent optimizer {k}: {e}")

            meta_path = os.path.join(mpath, "meta_agents")
            meta_agents_paths = glob.glob(os.path.join(meta_path, "*.agent"))
            
            for meta_agent_path in meta_agents_paths:
                try:
                    meta_agent = torch.load(meta_path)
                    meta_agent_id = meta_agent.id 
                    if meta_agent_id not in self.meta_agents.keys():
                        print(f"WARNING: loading a meta agent that was not there previously...: {meta_agent_id}.")
                    self.meta_agents[meta_agent_id] = meta_agent
                except Exception as e:
                    print(f"WARNING: exception caught when trying to load meta agent {meta_agent_id}: {e}")

    def _select_agents(self):
        idx_speaker = random.randint(0,len(self.speakers)-1)
        idx_listener = random.randint(0,len(self.listeners)-1)
            
        speaker = self.speakers[idx_speaker]
        listener = self.listeners[idx_listener]
        
        self.agents_stats[speaker.agent_id]['selection_iterations'].append(
            sum(self.previous_global_it_datasample.values())
        )
        self.agents_stats[listener.agent_id]['selection_iterations'].append(
            sum(self.previous_global_it_datasample.values())
        )
        
        return speaker, listener

    def bookkeeping(self, mode, epoch):
        it = sum(self.previous_global_it_datasample.values())
        
        if epoch != self.previous_epoch:
            self.previous_epoch = epoch
            # Save agent:
            for agent in self.speakers:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            for agent in self.listeners:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            
            if 'cultural_reset_strategy' in self.config\
            and 'meta' in self.config['cultural_reset_strategy']:
                for agent in self.meta_agents.values():
                    agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))

        # Reset agent(s):
        if 'train' in mode \
        and 'cultural_pressure_it_period' in self.config\
        and self.config["cultural_pressure_it_period"] is not None \
        and sum(self.previous_global_it_datasample.values()) % self.config['cultural_pressure_it_period'] == 0:
        #and (idx_stimuli+len(data_loader)*epoch) % self.config['cultural_pressure_it_period'] == 0:
            if 'oldest' in self.config['cultural_reset_strategy']:
                if 'S' in self.config['cultural_reset_strategy']:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.speakers] 
                    idx_speaker2reset = random.choices( range(len(self.speakers)), weights=weights)[0]
                if 'L' in self.config['cultural_reset_strategy']:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.listeners] 
                    idx_listener2reset = random.choices( range(len(self.listeners)), weights=weights)[0]
                else:
                    weights = [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.listeners] 
                    weights += [ it-self.agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in self.speakers]
                    idx_agent2reset = random.choices( range(len(self.listeners)+len(self.speakers)), weights=weights)[0]
            else: #uniform
                if 'S' in self.config['cultural_reset_strategy']:
                    idx_speaker2reset = random.randint(0,len(self.speakers)-1)
                if 'L' in self.config['cultural_reset_strategy']:
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
                
                if self.verbose:
                    print("Agent Speaker {} has just been resetted.".format(self.speakers[idx_speaker2reset].agent_id))
            
            if 'L' in self.config['cultural_reset_strategy']:
                if 'meta' in self.config['cultural_reset_strategy']:
                    self._apply_meta_update(meta_learner=self.meta_agents[type(self.listeners[idx_listener2reset])],
                                           meta_optimizer=self.meta_agents_optimizers[type(self.listeners[idx_listener2reset])],
                                           learner=self.listeners[idx_listener2reset])
                else:
                    self.listeners[idx_listener2reset].reset()
                self.agents_stats[self.listeners[idx_listener2reset].agent_id]['reset_iterations'].append(it)
                if self.verbose:
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
                
                if self.verbose:
                    print("Agents {} has just been resetted.".format(agents[idx_agent2reset].agent_id))
    
    def _reptile_step(self, learner, reptile_learner, nbr_grad_steps=1, verbose=False) :
        k = 1.0/float(nbr_grad_steps)
        nbrParams = 0
        nbrUpdatedParams = 0
        for (name, lp), (namer, lrp) in zip( learner.named_parameters(), reptile_learner.named_parameters() ) :
            nbrParams += 1
            if lrp.grad is not None:
                nbrUpdatedParams += 1
                lrp.grad.data.copy_( k*(lp.data.cpu()-lrp.data.cpu()) )
            else:
                lrp.grad = k*(lp.data.cpu()-lrp.data.cpu())
                if verbose:
                    print("Parameter {} has not been updated...".format(name))
        if verbose:
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
        
        if mode not in self.previous_global_it_datasample:
            self.previous_global_it_datasample[mode] = -1
            
        if global_it_datasample != self.previous_global_it_datasample[mode]:
            self.bookkeeping(mode=mode, epoch=epoch)

            self.previous_global_it_datasample[mode] = global_it_datasample
            
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

                if 'iterated_learning_scheme' in self.config\
                    and self.config['iterated_learning_scheme']\
                    and self.counterGames%self.config['iterated_learning_period']==0:
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
