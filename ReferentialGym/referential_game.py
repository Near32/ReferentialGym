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
from .networks import handle_nan


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
    def __init__(self, dataset, config):
        '''

        '''
        self.dataset = dataset
        self.config = config

    def train(self, prototype_speaker: Speaker, prototype_listener: Listener, nbr_epoch: int = 10, logger: SummaryWriter = None, verbose_period=None):
        '''

        '''
        # Dataset:
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
        if 'dataloader_num_worker' not in self.config:
            self.config['dataloader_num_worker'] = 8

        print("Create dataloader: ...")
        
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True,
                                                  num_workers=self.config['dataloader_num_worker'])

        print("Create dataloader: OK.")
        print("Create Agents: ...")
        
        # Agents:
        nbr_agents = self.config['cultural_substrate_size']
        speakers = [prototype_speaker]+[ copy.deepcopy(prototype_speaker) for i in range(nbr_agents-1)]
        listeners = [prototype_listener]+[ copy.deepcopy(prototype_listener) for i in range(nbr_agents-1)]
        speakers_optimizers = [ optim.Adam(speaker.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for speaker in speakers ]
        listeners_optimizers = [ optim.Adam(listener.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_eps']) for listener in listeners ]

        criterion = nn.CrossEntropyLoss(reduction='mean')

        print("Create Agents: OK.")
        print("Launching training: ...")

        for epoch in range(nbr_epoch):
            idx_speaker = random.randint(0,len(speakers)-1)
            idx_listener = random.randint(0,len(listeners)-1)
                
            speaker, speaker_optimizer = speakers[idx_speaker], speakers_optimizers[idx_speaker]
            listener, listener_optimizer = listeners[idx_listener], listeners_optimizers[idx_listener]
            
            if self.config['use_cuda']:
                speaker = speaker.cuda()
                listener = listener.cuda()
                
            for idx_stimuli, stimuli in enumerate(data_loader):
                speaker_optimizer.zero_grad()
                listener_optimizer.zero_grad()

                if self.config['use_cuda']:
                    stimuli = stimuli.cuda()

                shuffled_stimuli, target_decision_idx = shuffle(stimuli)
                speaker_stimuli = stimuli 
                if self.config['observability'] == "partial":
                    speaker_stimuli = speaker_stimuli[:,0].unsqueeze(1)

                listener_sentences_logits = None
                listener_sentences = None 

                for idx_round in range(self.config['nbr_communication_round']):
                    multi_round = True
                    if idx_round == self.config['nbr_communication_round']-1:
                        multi_round = False

                    speaker_sentences_logits, speaker_sentences = speaker(stimuli=speaker_stimuli, 
                                                                          sentences=listener_sentences, 
                                                                          graphtype=self.config['graphtype'], 
                                                                          tau0=self.config['tau0'], 
                                                                          multi_round=multi_round)
                    
                    decision_logits, listener_sentences_logits, listener_sentences = listener(sentences=speaker_sentences, 
                                                                                              stimuli=shuffled_stimuli, 
                                                                                              graphtype=self.config['graphtype'], 
                                                                                              tau0=self.config['tau0'],
                                                                                              multi_round=multi_round)
                    
                final_decision_logits = decision_logits
                if final_decision_logits.is_cuda: target_decision_idx = target_decision_idx.cuda()
                decision_probs = F.softmax( final_decision_logits, dim=-1)
                loss = criterion( decision_probs, target_decision_idx)

                # Weight MaxL1 Loss:
                weight_maxl1_loss = 0.0
                for p in speaker.parameters() :
                    weight_maxl1_loss += torch.max( torch.abs(p) )
                for p in listener.parameters() :
                    weight_maxl1_loss += torch.max( torch.abs(p) )
                
                if self.config['with_weight_maxl1_loss']:
                    loss += 0.5*weight_maxl1_loss

                loss.backward()
                
                nn.utils.clip_grad_value_(speaker.parameters(), self.config['gradient_clip'])
                nn.utils.clip_grad_value_(listener.parameters(), self.config['gradient_clip'])
                speaker.apply(handle_nan)
                listener.apply(handle_nan)

                speaker_optimizer.step()
                listener_optimizer.step()

                if logger is not None:
                    logger.add_scalar('Training/Loss', loss.item(), idx_stimuli+len(data_loader)*epoch)
                    logger.add_scalar('Training/WeightMaxL1Loss', weight_maxl1_loss.item(), idx_stimuli+len(data_loader)*epoch)
                    decision_idx = decision_probs.max(dim=-1)[1]
                    acc = (decision_idx==target_decision_idx).float().mean()
                    logger.add_scalar('Training/Accuracy', acc.item(), idx_stimuli+len(data_loader)*epoch)
                    
                    '''
                    for idx, sp in enumerate(speakers):
                        for name, p in sp.named_parameters() :
                            logger.add_histogram( "Speaker{}/{}".format(idx, name), p, idx_stimuli+len(data_loader)*epoch)
                            logger.add_histogram( "Speaker{}/{}/Grad".format(idx, name), p.grad, idx_stimuli+len(data_loader)*epoch)
                    for idx, ls in enumerate(listeners):
                        for name, p in ls.named_parameters() :
                            logger.add_histogram( "Listener{}/{}".format(idx, name), p, idx_stimuli+len(data_loader)*epoch)
                            logger.add_histogram( "Listener{}/{}/Grad".format(idx, name), p.grad, idx_stimuli+len(data_loader)*epoch)
                    '''
                
                if verbose_period is not None and idx_stimuli % verbose_period == 0:
                    print('Epoch {} :: Iteration {}/{} :: Training/Loss {} = {}'.format(epoch, idx_stimuli, len(data_loader), idx_stimuli+len(data_loader)*epoch, loss.item()))

                if self.config["cultural_pressure_it_period"] is not None and (idx_stimuli+len(data_loader)*epoch) % self.config['cultural_pressure_it_period'] == 0:
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




            



