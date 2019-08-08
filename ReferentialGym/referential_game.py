from typing import Dict, List, Tuple
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from .agents import Speaker, Listener
from .networks import handle_nan, hasnan


def shuffle(stimuli):
    st_size = stimuli.size()
    batch_size = st_size[0]
    nbr_distractors_po = st_size[1]
    perms = []
    shuffled_stimuli = []
    for b in range(batch_size):
        perm = torch.randperm(nbr_distractors_po)
        if stimuli.is_cuda: perm = perm.cuda()
        perms.append(perm.unsqueeze(0))
        shuffled_stimuli.append( stimuli[b,perm,...].unsqueeze(0))
    perms = torch.cat(perms, dim=0)
    shuffled_stimuli = torch.cat(shuffled_stimuli, dim=0)
    decision_target = (perms==0).max(dim=1)[1].long()
    return shuffled_stimuli, decision_target


class ReferentialGame(object):
    def __init__(self, datasets, config):
        '''

        '''
        self.datasets = datasets
        self.config = config

    def _select_agents(self, speakers, listeners, speakers_optimizers, listeners_optimizers):
        idx_speaker = random.randint(0,len(speakers)-1)
        idx_listener = random.randint(0,len(listeners)-1)
            
        speaker, speaker_optimizer = speakers[idx_speaker], speakers_optimizers[idx_speaker]
        listener, listener_optimizer = listeners[idx_listener], listeners_optimizers[idx_listener]
        
        if self.config['use_cuda']:
            speaker = speaker.cuda()
            listener = listener.cuda()
            
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
                                                            num_workers=self.config['dataloader_num_worker']),
                        'test':torch.utils.data.DataLoader(self.datasets['test'],
                                                           batch_size=self.config['batch_size'],
                                                           shuffle=True,
                                                           num_workers=self.config['dataloader_num_worker'])
                        }

        print("Create dataloader: OK.")
        print("Create Agents: ...")
        
        # Agents:
        nbr_agents = self.config['cultural_substrate_size']
        speakers = [prototype_speaker]+[ copy.deepcopy(prototype_speaker) for i in range(nbr_agents-1)]
        listeners = [prototype_listener]+[ copy.deepcopy(prototype_listener) for i in range(nbr_agents-1)]
        speakers_optimizers = [ optim.Adam(speaker.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for speaker in speakers ]
        listeners_optimizers = [ optim.Adam(listener.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for listener in listeners ]

        print("Create Agents: OK.")
        print("Launching training: ...")

        for epoch in range(nbr_epoch):
            for mode in ['train','test']:
                data_loader = data_loaders[mode]
                for idx_stimuli, stimuli in enumerate(data_loader):

                    speaker, listener, speaker_optimizer, listener_optimizer = self._select_agents(speakers,
                                                                                                  listeners,
                                                                                                  speakers_optimizers,
                                                                                                  listeners_optimizers)
                    if 'obverter' in self.config['graphtype'] and idx_stimuli%2==0:
                        # Let us exchange the speaker and listener:
                        speaker, listener, speaker_optimizer, listener_optimizer = listener, speaker, listener_optimizer, speaker_optimizer

                    '''
                    if mode == 'train': 
                        speaker.train()
                        listener.train()
                    else:
                        speaker.eval()
                        listener.eval()
                    '''

                    if self.config['use_cuda']:
                        stimuli = stimuli.cuda()

                    shuffled_stimuli, target_decision_idx = shuffle(stimuli)
                    speaker_stimuli = stimuli 
                    if self.config['observability'] == "partial":
                        speaker_stimuli = speaker_stimuli[:,0].unsqueeze(1)

                    listener_sentences = None 
                    
                    for idx_round in range(self.config['nbr_communication_round']):
                        multi_round = True
                        if idx_round == self.config['nbr_communication_round']-1:
                            multi_round = False

                        speaker_outputs = speaker(stimuli=speaker_stimuli, 
                                                  sentences=listener_sentences, 
                                                  graphtype=self.config['graphtype'], 
                                                  tau0=self.config['tau0'], 
                                                  multi_round=multi_round)
                        
                        if 'obverter' in self.config['graphtype']:
                            speaker_outputs['sentences'] = speaker_outputs['sentences'].detach()

                        listener_outputs = listener(sentences=speaker_outputs['sentences'], 
                                                    stimuli=shuffled_stimuli, 
                                                    graphtype=self.config['graphtype'], 
                                                    tau0=self.config['tau0'],
                                                    multi_round=multi_round)

                        if 'obverter' in self.config['graphtype']:
                            if isinstance(listener_outputs['sentences'], torch.Tensor):
                                listener_outputs['sentences'] = listener_outputs['sentences'].detach()

                        decision_logits = listener_outputs['decision']
                        listener_sentences = listener_outputs['sentences']

                    final_decision_logits = decision_logits
                    if final_decision_logits.is_cuda: target_decision_idx = target_decision_idx.cuda()
                    decision_probs = F.softmax( final_decision_logits, dim=-1)
                    
                    criterion = nn.CrossEntropyLoss(reduction='mean')
                    loss = criterion( final_decision_logits, target_decision_idx)
                    
                    '''
                    criterion = nn.MSELoss(reduction='mean')
                    loss = criterion( decision_probs, nn.functional.one_hot(target_decision_idx, num_classes=decision_probs.size(1)).float())
                    '''
                    
                    losses = []
                    for b in range(final_decision_logits.size(0)):
                        fd_target_l = final_decision_logits[b,target_decision_idx[b]]
                        bl = 0.0
                        for d in range(final_decision_logits.size(1)):
                            if d == target_decision_idx[b]: continue
                            el = 1-fd_target_l+final_decision_logits[b][d]
                            el = torch.max(torch.zeros_like(el),el)
                            bl = bl + el 
                        losses.append( bl.unsqueeze(0))
                    losses = torch.cat(losses,dim=0)
                    '''
                    loss = losses.mean()
                    '''
                    
                    # Weight MaxL1 Loss:
                    weight_maxl1_loss = 0.0
                    for p in speaker.parameters() :
                        weight_maxl1_loss += torch.max( torch.abs(p) )
                    for p in listener.parameters() :
                        weight_maxl1_loss += torch.max( torch.abs(p) )
                    
                    if self.config['with_weight_maxl1_loss']:
                        loss += 0.5*weight_maxl1_loss

                    if mode == 'train':
                        speaker_optimizer.zero_grad()
                        listener_optimizer.zero_grad()
                        loss.backward()
                        
                        nn.utils.clip_grad_value_(speaker.parameters(), self.config['gradient_clip'])
                        nn.utils.clip_grad_value_(listener.parameters(), self.config['gradient_clip'])
                        speaker.apply(handle_nan)
                        listener.apply(handle_nan)

                        speaker_optimizer.step()
                        listener_optimizer.step()
                    
                    if logger is not None:
                        logger.add_scalar('{}/Loss'.format(mode), loss.item(), idx_stimuli+len(data_loader)*epoch)
                        logger.add_scalar('{}/WeightMaxL1Loss'.format(mode), weight_maxl1_loss.item(), idx_stimuli+len(data_loader)*epoch)
                        decision_idx = decision_probs.max(dim=-1)[1]
                        acc = (decision_idx==target_decision_idx).float().mean()*100
                        logger.add_scalar('{}/Accuracy'.format(mode), acc.item(), idx_stimuli+len(data_loader)*epoch)
                        
                        logger.add_histogram( "{}/LossesPerStimulus".format(mode), losses, idx_stimuli+len(data_loader)*epoch)
                        logger.add_scalar( "{}/Stimulus/Mean".format(mode), stimuli.mean(), idx_stimuli+len(data_loader)*epoch)
                        
                        if mode == 'train':
                            if hasattr(speaker,'tau'): 
                                logger.add_histogram( "{}/Speaker/Tau".format(mode), speaker.tau, idx_stimuli+len(data_loader)*epoch)
                                logger.add_scalar( "{}/Tau/Speaker".format(mode), speaker.tau.mean().item(), idx_stimuli+len(data_loader)*epoch)
                            if hasattr(listener,'tau'): 
                                logger.add_histogram( "{}/Listener/Tau".format(mode), listener.tau, idx_stimuli+len(data_loader)*epoch)
                                logger.add_scalar( "{}/Tau/Listener".format(mode), listener.tau.mean().item(), idx_stimuli+len(data_loader)*epoch)
                        
                    if verbose_period is not None and idx_stimuli % verbose_period == 0:
                        print('Epoch {} :: {} Iteration {}/{} :: Training/Loss {} = {}'.format(epoch, mode, idx_stimuli, len(data_loader), idx_stimuli+len(data_loader)*epoch, loss.item()))

                    if mode == 'train' and self.config["cultural_pressure_it_period"] is not None and (idx_stimuli+len(data_loader)*epoch) % self.config['cultural_pressure_it_period'] == 0:
                        idx_speaker2reset = random.randint(0,len(speakers)-1)
                        idx_listener2reset = random.randint(0,len(listeners)-1)
                        
                        speakers[idx_speaker2reset].reset()
                        speakers_optimizers[idx_speaker2reset] = optim.Adam(speakers[idx_speaker2reset].parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps'])
                        listeners[idx_listener2reset].reset()
                        listeners_optimizers[idx_listener2reset] = optim.Adam(listeners[idx_listener2reset].parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps'])

                        print("Agents Speaker{} and Listener{} have just been resetted.".format(idx_speaker2reset, idx_listener2reset))



            # Save agent:
            torch.save(prototype_speaker, './basic_{}_speaker.pt'.format(prototype_speaker.kwargs['architecture']))
            torch.save(prototype_listener, './basic_{}_listener.pt'.format(prototype_listener.kwargs['architecture']))




            




