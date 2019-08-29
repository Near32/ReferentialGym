from typing import Dict, List, Tuple
import os
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from .agents import Speaker, Listener, ObverterAgent
from .networks import handle_nan, hasnan
from .datasets import collate_dict_wrapper
from .utils import cardinality


class ReferentialGame(object):
    def __init__(self, datasets, config):
        '''

        '''
        self.datasets = datasets
        self.config = config

    def _select_agents(self, it, speakers, listeners, speakers_optimizers, listeners_optimizers, agents_stats):
        idx_speaker = random.randint(0,len(speakers)-1)
        idx_listener = random.randint(0,len(listeners)-1)
            
        speaker, speaker_optimizer = speakers[idx_speaker], speakers_optimizers[idx_speaker]
        listener, listener_optimizer = listeners[idx_listener], listeners_optimizers[idx_listener]
        
        if self.config['use_cuda']:
            speaker = speaker.cuda()
            listener = listener.cuda()
        
        agents_stats[speaker.agent_id]['selection_iterations'].append(it)
        agents_stats[listener.agent_id]['selection_iterations'].append(it)

        return speaker, listener, speaker_optimizer, listener_optimizer

    def train(self, prototype_speaker: Speaker, prototype_listener: Listener, nbr_epoch: int = 10, logger: SummaryWriter = None, verbose_period=None):
        '''

        '''
        # Dataset:
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
        if 'dataloader_num_worker' not in self.config:
            self.config['dataloader_num_worker'] = 8

        print("Create dataloader: ...")
        
        data_loaders = {'train':torch.utils.data.DataLoader(self.datasets['train'],
                                                            batch_size=self.config['batch_size'],
                                                            shuffle=True,
                                                            collate_fn=collate_dict_wrapper,
                                                            pin_memory=True,
                                                            num_workers=self.config['dataloader_num_worker']),
                        'test':torch.utils.data.DataLoader(self.datasets['test'],
                                                           batch_size=self.config['batch_size'],
                                                           shuffle=True,
                                                           collate_fn=collate_dict_wrapper,
                                                           pin_memory=True,
                                                           num_workers=self.config['dataloader_num_worker'])
                        }

        print("Create dataloader: OK.")
        print("Create Agents: ...")
        
        # Agents:
        nbr_agents = self.config['cultural_substrate_size']
        speakers = [prototype_speaker]+[ prototype_speaker.clone(clone_id=f's{i+1}') for i in range(nbr_agents-1)]
        listeners = [prototype_listener]+[ prototype_listener.clone(clone_id=f'l{i+1}') for i in range(nbr_agents-1)]
        speakers_optimizers = [ optim.Adam(speaker.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for speaker in speakers ]
        listeners_optimizers = [ optim.Adam(listener.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for listener in listeners ]
        
        if 'meta' in self.config['cultural_reset_strategy']:
            meta_agents = dict()
            meta_agents_optimizers = dict()
            for agent in [prototype_speaker, prototype_listener]: 
                if type(agent) not in meta_agents:
                    meta_agents[type(agent)] = agent.clone(clone_id=f'meta_{agent.agent_id}')
                    #meta_agents_optimizers[type(agent)] = optim.Adam(meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'], eps=self.config['adam_eps'])
                    meta_agents_optimizers[type(agent)] = optim.SGD(meta_agents[type(agent)].parameters(), lr=self.config['cultural_reset_meta_learning_rate'])

        agents_stats = dict()
        for agent in speakers:
            agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        for agent in listeners:
            agents_stats[agent.agent_id] = {'reset_iterations':[0], 'selection_iterations':[]}
        
        print("Create Agents: OK.")
        print("Launching training: ...")

        it = 0
        for epoch in range(nbr_epoch):
            for mode in ['train','test']:
                data_loader = data_loaders[mode]
                counterGames = 0
                total_sentences = []
                total_nbr_unique_stimulus = 0
                for idx_stimuli, sample in enumerate(data_loader):
                    it += 1
                    if mode == 'train':
                        if'obverter' in self.config['graphtype']:
                            # Let us decide whether to exchange the speakers and listeners:
                            # i.e. is the round of games finished?
                            if not('obverter_nbr_games_per_round' in self.config):
                                self.config['obverter_nbr_games_per_round'] = 1 
                            counterGames += 1
                            if  counterGames%self.config['obverter_nbr_games_per_round']==0:
                                speakers, \
                                listeners, \
                                speakers_optimizers, \
                                listeners_optimizers = (listeners, \
                                                       speakers, \
                                                       listeners_optimizers, \
                                                       speakers_optimizers)
                            if self.config['iterated_learning_scheme'] and counterGames%self.config['iterated_learning_period']==0:
                                for lidx in range(len(listeners)):
                                    listeners[lidx].reset()
                                    print("Iterated Learning Scheme: Listener {} have just been resetted.".format(listeners[lidx].agent_id))
                                    listeners_optimizers[lidx] = optim.Adam(listeners[lidx].parameters(), 
                                        lr=self.config['learning_rate'], 
                                        eps=self.config['adam_eps'])

                    speaker, \
                    listener, \
                    speaker_optimizer, \
                    listener_optimizer = self._select_agents(it,
                                                             speakers,
                                                             listeners,
                                                             speakers_optimizers,
                                                             listeners_optimizers,
                                                             agents_stats)
                    '''
                    if mode == 'train': 
                        speaker.train()
                        listener.train()
                    else:
                        speaker.eval()
                        listener.eval()
                    '''

                    if self.config['use_cuda']:
                        sample = sample.cuda()

                    listener_sentences = None 
                    batch_size = len(sample['speaker_experiences'])

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    
                        
                    for idx_round in range(self.config['nbr_communication_round']):
                        multi_round = True
                        if idx_round == self.config['nbr_communication_round']-1:
                            multi_round = False
                        
                        speaker_inputs_dict = {'experiences':sample['speaker_experiences'], 
                                               'sentences':listener_sentences, 
                                               'graphtype':self.config['graphtype'],
                                               'tau0':self.config['tau0'],
                                               'multi_round':multi_round,
                                               'sample':sample
                                               }
                        speaker_outputs, speaker_losses = speaker.train(inputs_dict=speaker_inputs_dict,
                                                                        config=self.config,
                                                                        role='speaker')

                        
                        listener_inputs_dict = {'graphtype':self.config['graphtype'],
                                                'tau0':self.config['tau0'],
                                                'multi_round':multi_round,
                                                'sample':sample}

                        for k in speaker_outputs:
                            listener_inputs_dict[k] = speaker_outputs[k] 

                        if self.config['graphtype'] == 'obverter':
                            if isinstance(speaker_outputs['sentences'], torch.Tensor):
                                listener_inputs_dict['sentences'] = listener_inputs_dict['sentences'].detach()
                            if isinstance(speaker_outputs['sentences_widx'], torch.Tensor):
                                listener_inputs_dict['sentences_widx'] = listener_inputs_dict['sentences_widx'].detach()

                        listener_inputs_dict['experiences'] = sample['listener_experiences']
                        listener_inputs_dict['sentences'] = listener_inputs_dict['sentences_widx']

                        listener_outputs, listener_losses = listener.train(inputs_dict=listener_inputs_dict,
                                                                           config=self.config,
                                                                           role='listener')

                        if self.config['graphtype'] == 'obverter':
                            if isinstance(listener_outputs['sentences'], torch.Tensor):
                                listener_outputs['sentences'] = listener_outputs['sentences'].detach()
                            if isinstance(listener_outputs['sentences_widx'], torch.Tensor):
                                listener_outputs['sentences_widx'] = listener_outputs['sentences_widx'].detach()

                        listener_sentences = listener_outputs['sentences_widx']

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    
                    final_decision_logits = listener_outputs['decision']
                    
                    losses = dict()
                    losses.update(speaker_losses)
                    losses.update(listener_losses)
                    for k, v in losses.items():
                        losses[k][1] = v[0]*v[1]
                    loss = sum( [l[1] for l in losses.values()])

                    if mode == 'train':
                        speaker_optimizer.zero_grad()
                        listener_optimizer.zero_grad()
                        loss.backward()
                        
                        speaker.apply(handle_nan)
                        listener.apply(handle_nan)
                        if self.config['with_gradient_clip']:
                            nn.utils.clip_grad_value_(speaker.parameters(), self.config['gradient_clip'])
                            nn.utils.clip_grad_value_(listener.parameters(), self.config['gradient_clip'])
                        
                        speaker_optimizer.step()
                        listener_optimizer.step()
                    

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    
                    # Compute sentences:
                    sentences = []
                    for sidx in range(batch_size):
                        sentences.append(''.join([chr(97+int(s.item())) for s in speaker_outputs['sentences_widx'][sidx] ]))
                    
                    if logger is not None:
                        if self.config['with_grad_logging']:
                            maxgrad = 0.0
                            for name, p in speaker.named_parameters() :
                                if hasattr(p,'grad') and p.grad is not None:
                                    logger.add_histogram( "Speaker/{}".format(name), p.grad, idx_stimuli+len(data_loader)*epoch)
                                    cmg = torch.abs(p.grad).max()
                                    if cmg > maxgrad:
                                        maxgrad = cmg
                            logger.add_scalar( "{}/SpeakerMaxGrad".format(mode), maxgrad, idx_stimuli+len(data_loader)*epoch)                    
                            maxgrad = 0
                            for name, p in listener.named_parameters() :
                                if hasattr(p,'grad') and p.grad is not None:
                                    logger.add_histogram( "Listener/{}".format(name), p.grad, idx_stimuli+len(data_loader)*epoch)                    
                                    cmg = torch.abs(p.grad).max()
                                    if cmg > maxgrad:
                                        maxgrad = cmg
                            logger.add_scalar( "{}/ListenerMaxGrad".format(mode), maxgrad, idx_stimuli+len(data_loader)*epoch)                    
                        
                        if self.config['with_utterance_penalization'] or self.config['with_utterance_promotion']:
                            for widx in range(self.config['vocab_size']+1):
                                logger.add_scalar("{}/Word{}Counts".format(mode,widx), speaker_outputs['speaker_utterances_count'][widx], idx_stimuli+len(data_loader)*epoch)
                            logger.add_scalar("{}/OOVLoss".format(mode), speaker_losses['oov_loss'][0], idx_stimuli+len(data_loader)*epoch)

                        if self.config['with_mdl_principle']:
                            logger.add_scalar("{}/MDLLoss".format(mode), speaker_losses['mdl_loss'][0], idx_stimuli+len(data_loader)*epoch)
                        
                        #sentence_length = sum([ float(s.size(1)) for s in speaker_outputs['sentences_widx']])/len(speaker_outputs['sentences_widx'])
                        sentence_length = (speaker_outputs['sentences_widx']<self.config['vocab_size']).sum().float()/batch_size
                        logger.add_scalar('{}/SentenceLength (/{})'.format(mode, self.config['max_sentence_length']), sentence_length/self.config['max_sentence_length'], idx_stimuli+len(data_loader)*epoch)
                        
                        for sentence in sentences:  total_sentences.append(sentence.replace(chr(97+self.config['vocab_size']), ''))
                        total_nbr_unique_sentences = cardinality(total_sentences)
                        total_nbr_unique_stimulus += batch_size
                        logger.add_scalar('{}/Ambiguity (%)'.format(mode), float(total_nbr_unique_stimulus-total_nbr_unique_sentences)/total_nbr_unique_stimulus*100.0, idx_stimuli+len(data_loader)*epoch)
                        
                        entropies = torch.cat([torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy() for w_logits in s_logits], dim=0) for s_logits in speaker_outputs['sentences_logits']], dim=0)
                        logger.add_scalar('{}/Entropy'.format(mode), entropies.mean(), idx_stimuli+len(data_loader)*epoch)
                        
                        logger.add_scalar('{}/Loss'.format(mode), loss.item(), idx_stimuli+len(data_loader)*epoch)
                        logger.add_scalar('{}/WeightMaxL1Loss'.format(mode), speaker_outputs['maxl1_loss'].item()+listener_outputs['maxl1_loss'].item(), idx_stimuli+len(data_loader)*epoch)
                        
                        decision_probs = listener_outputs['decision_probs']

                        if self.config['descriptive']:
                            if self.config['obverter_least_effort_loss']:
                                for widx in range(len(losses4widx)):
                                    logger.add_scalar('{}/Loss@w{}'.format(mode,widx), losses4widx[widx].item(), idx_stimuli+len(data_loader)*epoch)
                                    acc = (torch.abs(decision_probs[:,widx,...]-target_decision_probs) < 0.5).float().mean()*100
                                    logger.add_scalar('{}/Accuracy@{}'.format(mode,widx), acc.item(), idx_stimuli+len(data_loader)*epoch)
                            else:
                                #acc = (torch.abs(decision_probs-target_decision_probs) < 0.5).float().mean()*100
                                decision_idx = decision_probs.max(dim=-1)[1]
                                acc = (decision_idx==sample['target_decision_idx']).float().mean()*100
                        else:
                            decision_idx = decision_probs.max(dim=-1)[1]
                            acc = (decision_idx==sample['target_decision_idx']).float().mean()*100
                        logger.add_scalar('{}/Accuracy'.format(mode), acc.item(), idx_stimuli+len(data_loader)*epoch)
                        
                        #logger.add_histogram( "{}/LossesPerStimulus".format(mode), losses, idx_stimuli+len(data_loader)*epoch)
                        logger.add_scalar( "{}/Stimulus/Mean".format(mode), sample['listener_experiences'].mean(), idx_stimuli+len(data_loader)*epoch)
                        
                        if mode == 'train':
                            if hasattr(speaker,'tau'): 
                                logger.add_histogram( "{}/Speaker/Tau".format(mode), speaker.tau, idx_stimuli+len(data_loader)*epoch)
                                logger.add_scalar( "{}/Tau/Speaker".format(mode), speaker.tau.mean().item(), idx_stimuli+len(data_loader)*epoch)
                            if hasattr(listener,'tau'): 
                                logger.add_histogram( "{}/Listener/Tau".format(mode), listener.tau, idx_stimuli+len(data_loader)*epoch)
                                logger.add_scalar( "{}/Tau/Listener".format(mode), listener.tau.mean().item(), idx_stimuli+len(data_loader)*epoch)
                    

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    

                    if verbose_period is not None and idx_stimuli % verbose_period == 0:
                        print('Epoch {} :: {} Iteration {}/{} :: Loss {} = {}'.format(epoch, mode, idx_stimuli, len(data_loader), idx_stimuli+len(data_loader)*epoch, loss.item()))
                        # Show some sentences:
                        for sidx in range(batch_size):
                            if sidx>=10:
                                break
                            print(f"{sidx} : ", sentences[sidx], \
                                f"\t /label: {sample['target_decision_idx'][sidx]}",\
                                f" /decision: {decision_idx[sidx]}")
                    

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//

                    
                    if mode == 'train' and self.config["cultural_pressure_it_period"] is not None and (idx_stimuli+len(data_loader)*epoch) % self.config['cultural_pressure_it_period'] == 0:
                        if self.config['use_cuda']:
                            speaker = speaker.cpu()
                            listener = listener.cpu()

                        if 'oldest' in self.config['cultural_reset_strategy']:
                            if 'S' in self.config['cultural_reset_strategy']:
                                weights = [ it-agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in speakers ] 
                                idx_speaker2reset = random.choices( range(len(speakers)), weights=weights)[0]
                            elif 'L' in self.config['cultural_reset_strategy']:
                                weights = [ it-agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in listeners ] 
                                idx_listener2reset = random.choices( range(len(listeners)), weights=weights)[0]
                            else:
                                weights = [ it-agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in listeners ] 
                                weights += [ it-agents_stats[agent.agent_id]['reset_iterations'][-1] for agent in speakers ]
                                idx_agent2reset = random.choices( range(len(listeners)*2), weights=weights)[0]
                        else: #uniform
                            if 'S' in self.config['cultural_reset_strategy']:
                                idx_speaker2reset = random.randint(0,len(speakers)-1)
                            elif 'L' in self.config['cultural_reset_strategy']:
                                idx_listener2reset = random.randint(0,len(listeners)-1)
                            else:
                                idx_agent2reset = random.randint(0,2*len(listeners)-1)

                        if 'S' in self.config['cultural_reset_strategy']:
                            if 'meta' in self.config['cultural_reset_strategy']:
                                self._apply_meta_update(meta_learner=meta_agents[type(speakers[idx_speaker2reset])],
                                                       meta_optimizer=meta_agents_optimizers[type(speakers[idx_speaker2reset])],
                                                       learner=speakers[idx_speaker2reset])
                            else:
                                speakers[idx_speaker2reset].reset()
                            speakers_optimizers[idx_speaker2reset] = optim.Adam(speakers[idx_speaker2reset].parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps'])
                            agents_stats[speakers[idx_speaker2reset].agent_id]['reset_iterations'].append(it)
                            print("Agent Speaker {} has just been resetted.".format(speakers[idx_speaker2reset].agent_id))
                        elif 'L' in self.config['cultural_reset_strategy']:
                            if 'meta' in self.config['cultural_reset_strategy']:
                                self._apply_meta_update(meta_learner=meta_agents[type(listeners[idx_listener2reset])],
                                                       meta_optimizer=meta_agents_optimizers[type(listeners[idx_listener2reset])],
                                                       learner=listeners[idx_listener2reset])
                            else:
                                listeners[idx_listener2reset].reset()
                            listeners_optimizers[idx_listener2reset] = optim.Adam(listeners[idx_listener2reset].parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps'])
                            agents_stats[listeners[idx_listener2reset].agent_id]['reset_iterations'].append(it)
                            print("Agent  Listener {} has just been resetted.".format(speakers[idx_listener2reset].agent_id))
                        else:
                            if idx_agent2reset < len(listeners):
                                agents = listeners 
                                agents_optimizers = listeners_optimizers
                            else:
                                agents = speakers 
                                agents_optimizers = speakers_optimizers
                                idx_agent2reset -= len(listeners)
                            if 'meta' in self.config['cultural_reset_strategy']:
                                self._apply_meta_update(meta_learner=meta_agents[type(agents[idx_agent2reset])],
                                                       meta_optimizer=meta_agents_optimizers[type(agents[idx_agent2reset])],
                                                       learner=agents[idx_agent2reset])
                            else:
                                agents[idx_agent2reset].reset()
                                agents_optimizers[idx_agent2reset] = optim.Adam(agents[idx_agent2reset].parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps'])
                                agents_stats[agents[idx_agent2reset].agent_id]['reset_iterations'].append(it)
                            print("Agents {} has just been resetted.".format(agents[idx_agent2reset].agent_id))
                            
                                
                    # //------------------------------------------------------------//
                    
                if logger is not None:
                    logger.switch_epoch()

            # Save agent:
            for agent in speakers:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            for agent in listeners:
                agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            
            if 'meta' in self.config['cultural_reset_strategy']:
                for agent in meta_agents.values():
                    agent.save(path=os.path.join(self.config['save_path'],'{}_{}.pt'.format(agent.kwargs['architecture'], agent.agent_id)))
            
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




            




