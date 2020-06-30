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
from tqdm import tqdm

from .agents import Speaker, Listener, ObverterAgent
from .networks import handle_nan, hasnan

from .datasets import collate_dict_wrapper
from .utils import cardinality, query_vae_latent_space

from .utils import StreamHandler

VERBOSE = False 


class ReferentialGame(object):
    def __init__(self, datasets, config, modules, pipelines):
        '''

        '''
        self.datasets = datasets
        self.config = config
        
        self.stream_handler = StreamHandler()
        self.stream_handler.register("current_listener")
        self.stream_handler.register("current_speaker")
        
        self.stream_handler.register("losses_dict")
        self.stream_handler.register("logs_dict")
        
        # Register hyperparameters:
        for k,v in self.config.items():
            self.stream_handler.update(f"config:{k}", v)
        # Register modules:
        self.modules = modules
        for k,m in self.modules.items():
            self.stream_handler.update(f"modules:{m.get_id()}:ref", m)

        # Register pipelines:
        self.pipelines = pipelines

    def train(self, nbr_epoch: int = 10, logger: SummaryWriter = None, verbose_period=None):
        '''

        '''
        # Dataset:
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
        if 'dataloader_num_worker' not in self.config:
            self.config['dataloader_num_worker'] = 8

        print("Create dataloader: ...")
        
        data_loaders = {mode:torch.utils.data.DataLoader(dataset,
                                                            batch_size=self.config['batch_size'],
                                                            shuffle=True,
                                                            collate_fn=collate_dict_wrapper,
                                                            pin_memory=True,
                                                            num_workers=self.config['dataloader_num_worker'])
                        for mode, dataset in self.datasets.items()
                        }
        
        print("Create dataloader: OK.")
        
        print("Launching training: ...")

        it = 0
        it_datasamples = {mode:0 for mode in self.datasets} # counting the number of data sampled from dataloaders
        it_samples = {mode:0 for mode in self.datasets} # counting the number of multi-round
        it_steps = {mode:0 for mode in self.datasets} # taking into account multi round... counting the number of sample shown to the agents.
        
        if 'use_curriculum_nbr_distractors' in self.config\
            and self.config['use_curriculum_nbr_distractors']:
            windowed_accuracy = 0.0
            window_count = 0
            for mode in self.datasets:
                self.datasets[mode].setNbrDistractors(1,mode=mode)
            
        pbar = tqdm(total=nbr_epoch)
        if logger is not None:
            self.stream_handler.update("modules:logger:ref", logger)
        
        for epoch in range(nbr_epoch):
            self.stream_handler.update("signals:epoch", epoch)
            pbar.update(1)
            for it_dataset, (mode, data_loader) in enumerate(data_loaders.items()):
                self.stream_handler.update("current_dataset:ref", self.datasets[mode])
                self.stream_handler.update("signals:mode", mode)
                counterGames = 0
                total_sentences = []
                total_nbr_unique_stimulus = 0
                epoch_acc = []

                end_of_epoch_dataset = (it_dataset==len(data_loaders)-1)
                self.stream_handler.update("signals:end_of_epoch_dataset", end_of_epoch_dataset)
                
                nbr_experience_repetition = 1
                if 'nbr_experience_repetition' in self.config\
                    and 'train' in mode:
                    nbr_experience_repetition = self.config['nbr_experience_repetition']

                for idx_stimulus, sample in enumerate(data_loader):
                    end_of_dataset = (idx_stimulus==len(data_loader)-1)
                    self.stream_handler.update("signals:end_of_dataset", end_of_dataset)
                    it_datasamples[mode] += 1
                    it_datasample = it_datasamples[mode]
                    self.stream_handler.update("signals:global_it_datasample", it_datasample)
                    self.stream_handler.update("signals:it_datasample", idx_stimulus)
                    it = it_datasample


                    if self.config['use_cuda']:
                        sample = sample.cuda()

                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    
                    for it_rep in range(nbr_experience_repetition):
                        it_samples[mode] += 1
                        it_sample = it_samples[mode]
                        self.stream_handler.update("signals:global_it_sample", it_sample)
                        self.stream_handler.update("signals:it_sample", it_rep)
                        end_of_repetition_sequence = (it_rep==nbr_experience_repetition-1)
                        self.stream_handler.update("signals:end_of_repetition_sequence", end_of_repetition_sequence)
                        
                        batch_size = len(sample['speaker_experiences'])
                        
                        # TODO: implement a multi_round_communicatioin module ?
                        for idx_round in range(self.config['nbr_communication_round']):
                            it_steps[mode] += 1
                            it_step = it_steps[mode]
                            
                            self.stream_handler.update("signals:global_it_step", it_step)
                            self.stream_handler.update("signals:it_step", idx_round)
                            
                            end_of_communication = (idx_round==self.config['nbr_communication_round']-1)
                            self.stream_handler.update("signals:end_of_communication", end_of_communication)
                            
                            multi_round = True
                            if end_of_communication:
                                multi_round = False
                            self.stream_handler.update("signals:multi_round", multi_round)
                            self.stream_handler.update('current_dataloader:sample', sample)

                            for pipe_id, pipeline in self.pipelines.items():
                                if "referential_game" in pipe_id: 
                                    self.stream_handler.serve(pipeline)

                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        
                        for pipe_id, pipeline in self.pipelines.items():
                            if "referential_game" not in pipe_id:
                                self.stream_handler.serve(pipeline)
                        
                        
                        losses = self.stream_handler["losses_dict"]
                        loss = sum( [l[-1] for l in losses.values()])
                        logs_dict = self.stream_handler["logs_dict"]
                        acc_keys = [k for k in logs_dict.keys() if '/referential_game_accuracy' in k]
                        if len(acc_keys):
                            acc = logs_dict[acc_keys[-1]].mean()

                        if verbose_period is not None and idx_stimulus % verbose_period == 0:
                            descr = 'Epoch {} :: {} Iteration {}/{} :: Loss {} = {}'.format(epoch+1, mode, idx_stimulus+1, len(data_loader), it+1, loss.item())
                            pbar.set_description_str(descr)
                        
                        self.stream_handler.reset("losses_dict")
                        self.stream_handler.reset("logs_dict")

                        '''
                        if logger is not None:
                            if self.config['with_utterance_penalization'] or self.config['with_utterance_promotion']:
                                import ipdb; ipdb.set_trace()
                                for widx in range(self.config['vocab_size']+1):
                                    logger.add_scalar("{}/Word{}Counts".format(mode,widx), speaker_outputs['speaker_utterances_count'][widx], it_step)
                                logger.add_scalar("{}/OOVLoss".format(mode), speaker_losses['oov_loss'][-1].mean().item(), it_step)
                            
                            if 'with_mdl_principle' in self.config and self.config['with_mdl_principle']:
                                logger.add_scalar("{}/MDLLoss".format(mode), speaker_losses['mdl_loss'][-1].mean().item(), it_step)
                        '''    
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        
                        # TODO: CURRICULUM ON DISTRATORS as a module that handles the current dataloader reference....!!
                        if 'use_curriculum_nbr_distractors' in self.config\
                            and self.config['use_curriculum_nbr_distractors']:
                            nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
                            logger.add_scalar( "{}/CurriculumNbrDistractors".format(mode), nbr_distractors, it_step)
                            logger.add_scalar( "{}/CurriculumWindowedAcc".format(mode), windowed_accuracy, it_step)
                        
                        
                        # TODO: make this a logger module:
                        if 'current_speaker' in self.modules and 'current_listener' in self.modules:
                            prototype_speaker = self.stream_handler["modules:current_speaker:ref_agent"]
                            prototype_listener = self.stream_handler["modules:current_listener:ref_agent"]
                            image_save_path = logger.path 
                            if prototype_speaker is not None and hasattr(prototype_speaker,'VAE') and idx_stimulus % 4 == 0:
                                query_vae_latent_space(prototype_speaker.VAE, 
                                                       sample=sample['speaker_experiences'],
                                                       path=image_save_path,
                                                       test=('test' in mode),
                                                       full=('test' in mode),
                                                       idxoffset=it_rep+idx_stimulus*self.config['nbr_experience_repetition'],
                                                       suffix='speaker',
                                                       use_cuda=True)
                                
                            if prototype_listener is not None and hasattr(prototype_listener,'VAE') and idx_stimulus % 4 == 0:
                                query_vae_latent_space(prototype_listener.VAE, 
                                                       sample=sample['listener_experiences'],
                                                       path=image_save_path,
                                                       test=('test' in mode),
                                                       full=('test' in mode),
                                                       idxoffset=idx_stimulus,
                                                       suffix='listener')
                                
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//
                    # //------------------------------------------------------------//

                    # TODO: many parts everywhere, do not forget them all : CURRICULUM ON DISTRACTORS...!!!
                    if 'train' in mode\
                        and 'use_curriculum_nbr_distractors' in self.config\
                        and self.config['use_curriculum_nbr_distractors']:
                        nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
                        windowed_accuracy = (windowed_accuracy*window_count+acc.item())
                        window_count += 1
                        windowed_accuracy /= window_count
                        if windowed_accuracy > 75 and window_count > self.config['curriculum_distractors_window_size'] and nbr_distractors < self.config['nbr_distractors'][mode]:
                            windowed_accuracy = 0
                            window_count = 0
                            for mode in self.datasets:
                                self.datasets[mode].setNbrDistractors(self.datasets[mode].getNbrDistractors(mode=mode)+1, mode=mode)
                    # //------------------------------------------------------------//

                if logger is not None:
                    logger.switch_epoch()

                # //------------------------------------------------------------//
                # //------------------------------------------------------------//
                # //------------------------------------------------------------//

            # //------------------------------------------------------------//
            # //------------------------------------------------------------//
            # //------------------------------------------------------------//

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------// 
        
        return



            




