from typing import Dict, List, Tuple
import os
import copy
import random
import time
import cloudpickle 
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm

from .agents import Speaker, Listener, ObverterAgent
from .networks import handle_nan, hasnan

from .datasets import collate_dict_wrapper, PrioritizedBatchSampler, PrioritizedSampler
from .utils import cardinality, query_vae_latent_space

from .utils import StreamHandler

VERBOSE = False 


class ReferentialGame(object):
    def __init__(self, 
                 datasets, 
                 config={}, 
                 modules={}, 
                 pipelines={}, 
                 load_path=None, 
                 save_path=None,
                 verbose=False,
                 use_priority=False,
                 save_epoch_interval=None):
        '''

        '''
        self.use_priority = use_priority
        self.verbose = verbose
        self.save_epoch_interval = save_epoch_interval

        self.load_path= load_path
        self.save_path = save_path

        self.datasets = datasets
        self.config = config
        if load_path is not None:
            self.load_config(load_path)
        
        self.stream_handler = StreamHandler()
        self.stream_handler.register("losses_dict")
        self.stream_handler.register("logs_dict")
        self.stream_handler.register("signals")
        if load_path is not None:
            self.load_signals(load_path)
        
        # Register hyperparameters:
        for k,v in self.config.items():
            self.stream_handler.update(f"config:{k}", v)
        # Register modules:
        self.modules = modules
        if load_path is not None:
            self.load_modules(load_path)
        for k,m in self.modules.items():
            self.stream_handler.update(f"modules:{m.get_id()}:ref", m)

        # Register pipelines:
        self.pipelines = pipelines
        if load_path is not None:
            self.load_pipelines(load_path)

    def save(self, path=None):
        if path is None:
            print("WARNING: no path provided for save. Saving in './temp_save/'.")
            path = './temp_save/'

        os.makedirs(path, exist_ok=True)

        self.save_config(path)
        self.save_modules(path)
        self.save_pipelines(path)
        self.save_signals(path)

        if self.verbose:
            print(f"Saving at {path}: OK.")

    def save_config(self, path):
        try:
            with open(os.path.join(path, "config.conf"), 'wb') as f:
                cloudpickle.dump(self.config, f) # protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save config: {e}")

    def save_modules(self, path):
        for module_id, module in self.modules.items():
            #try:
            if hasattr(module, "save"):
                module.save(path=path)
            else:
                #torch.save(module, os.path.join(path,module_id+".pth"))
                # TODO: find a better way to save modules with lambda functions...
                # maybe remove lambda functions altogether...
                try:
                    with open(os.path.join(path, module_id+".pth"), 'wb') as f:
                        cloudpickle.dump(module, f) # protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Exception caught while trying to save module {module_id}: {e}")
            #except Exception as e:
            #    print(f"Exception caught will trying to save module {module_id}: {e}")
                 

    def save_pipelines(self, path):
        try:
            with open(os.path.join(path, "pipelines.pipe"), 'wb') as f:
                cloudpickle.dump(self.pipelines, f) # protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save pipelines: {e}")

    def save_signals(self, path):
        try:
            with open(os.path.join(path, "signals.conf"), 'wb') as f:
                cloudpickle.dump(self.stream_handler["signals"], f) # protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Exception caught while trying to save signals: {e}")

    def load(self, path):
        self.load_config(path)
        self.load_modules(path)
        self.load_pipelines(path)
        self.load_signals(path)

        if self.verbose:
            print(f"Loading from {path}: OK.")


    def load_config(self, path):
        try:
            with open(os.path.join(path, "config.conf"), 'rb') as f:
                self.config = cloudpickle.load(f)
        except Exception as e:
            print(f"Exception caught while trying to load config: {e}")

        if self.verbose:
            print(f"Loading config: OK.")

    def load_modules(self, path):
        modules_paths = glob.glob(os.path.join(path, "*.pth"))
        
        for module_path in modules_paths:
            module_id = module_path.split("/")[-1].split(".")[0]
            try:
                #self.modules[module_id] = torch.load(module_path)
                with open(module_path, 'rb') as module_fle:
                    self.modules[module_id] = cloudpickle.load(module_file)
            except Exception as e:
                print(f"Exception caught will trying to load module {module_path}: {e}")
        
        if self.verbose:
            print(f"Loading modules: OK.")
    
    def load_pipelines(self, path):
        try:
            with open(os.path.join(path, "pipelines.pipe"), 'rb') as f:
                self.pipelines.update(cloudpickle.load(f))
        except Exception as e:
            print(f"Exception caught while trying to load pipelines: {e}")

        if self.verbose:
            print(f"Loading pipelines: OK.")

    def load_signals(self, path):
        try:
            with open(os.path.join(path, "signals.conf"), 'rb') as f:
                self.stream_handler.update("signals", cloudpickle.load(f))
        except Exception as e:
            print(f"Exception caught while trying to load signals: {e}")

        if self.verbose:
            print(f"Loading signals: OK.")

    def train(self, nbr_epoch: int = 10, logger: SummaryWriter = None, verbose_period=None):
        '''

        '''
        # Dataset:
        if 'batch_size' not in self.config:
            self.config['batch_size'] = 32
        if 'dataloader_num_worker' not in self.config:
            self.config['dataloader_num_worker'] = 8
        
        effective_batch_size = self.config['batch_size']
        from .datasets.dataset import DC_version
        if DC_version == 2:
            effective_batch_size = effective_batch_size // 2
        #print("Create dataloader: ...")
        
        data_loaders = {}
        self.pbss = {}
        for mode, dataset in self.datasets.items():
            if 'train' in mode and self.use_priority:
                capacity = len(dataset)
                
                """
                pbs = PrioritizedBatchSampler(
                    capacity=capacity,
                    batch_size=self.config['batch_size']
                )
                self.pbss[mode] = pbs
                data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    #batch_size=self.config['batch_size'],
                    #shuffle=True,
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker'],
                    batch_sampler=pbs,
                )

                """
                pbs = PrioritizedSampler(
                    capacity=capacity,
                    batch_size=effective_batch_size, #self.config['batch_size'],
                    logger=logger,
                )
                self.pbss[mode] = pbs
                data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=effective_batch_size, #self.config['batch_size'],
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker'],
                    sampler=pbs,
                )
            else:
                data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=effective_batch_size, #self.config['batch_size'],
                    shuffle=True,
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker']
                )
            
        #print("Create dataloader: OK (WARNING: num_worker arg disabled...).")
        
        #print("Launching training: ...")

        it_datasamples = self.stream_handler['signals:it_datasamples']
        if it_datasamples is None:  it_datasamples = {mode:0 for mode in self.datasets} # counting the number of data sampled from dataloaders
        it_samples = self.stream_handler['signals:it_samples']
        if it_samples is None:  it_samples = {mode:0 for mode in self.datasets} # counting the number of multi-round
        it_steps = self.stream_handler['signals:it_steps']
        if it_steps is None:    it_steps = {mode:0 for mode in self.datasets} # taking into account multi round... counting the number of sample shown to the agents.
        
        init_curriculum_nbr_distractors = self.stream_handler["signals:curriculum_nbr_distractors"]
        if init_curriculum_nbr_distractors is None:
            init_curriculum_nbr_distractors = self.config.get('init_curriculum_nbr_distractors', 1)
        if 'use_curriculum_nbr_distractors' in self.config\
            and self.config['use_curriculum_nbr_distractors']:
            windowed_accuracy = 0.0
            window_count = 0
            for mode in self.datasets:
                self.datasets[mode].setNbrDistractors(init_curriculum_nbr_distractors,mode=mode)
            
        if logger is not None:
            self.stream_handler.update("modules:logger:ref", logger)
        
        self.stream_handler.update("signals:use_cuda", self.config['use_cuda'])
        update_count = self.stream_handler["signals:update_count"]
        if update_count is None:
            self.stream_handler.update("signals:update_count", 0)
                
        init_epoch = self.stream_handler["signals:epoch"]
        if init_epoch is None: 
            init_epoch = 0
            print(f'WARNING: Referential Game : REINITIALIZED EPOCH COUNT.')
            pbar = tqdm(total=nbr_epoch)
        else:
            pbar = tqdm(total=init_epoch+nbr_epoch)
            pbar.update(init_epoch-1)

        for epoch in range(init_epoch,init_epoch+nbr_epoch):
            self.stream_handler.update("signals:epoch", epoch)
            pbar.update(1)
            for it_dataset, (mode, data_loader) in enumerate(data_loaders.items()):
                self.stream_handler.update("current_dataset:ref", self.datasets[mode])
                self.stream_handler.update("signals:mode", mode)
                
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
                    self.stream_handler.update("signals:it_datasamples", it_datasamples)
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
                        self.stream_handler.update("signals:it_samples", it_samples)
                        self.stream_handler.update("signals:global_it_sample", it_sample)
                        self.stream_handler.update("signals:it_sample", it_rep)
                        end_of_repetition_sequence = (it_rep==nbr_experience_repetition-1)
                        self.stream_handler.update("signals:end_of_repetition_sequence", end_of_repetition_sequence)
                        
                        # TODO: implement a multi_round_communicatioin module ?
                        for idx_round in range(self.config['nbr_communication_round']):
                            it_steps[mode] += 1
                            it_step = it_steps[mode]
                            
                            self.stream_handler.update("signals:it_steps", it_steps)
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

                        if self.use_priority and mode in self.pbss:
                            batched_loss = sum([l for l in losses.values()]).detach().cpu().numpy()
                            self.pbss[mode].update_batch(batched_loss)

                        loss = sum( [l.mean() for l in losses.values()])
                        logs_dict = self.stream_handler["logs_dict"]
                        acc_keys = [k for k in logs_dict.keys() if '/referential_game_accuracy' in k]
                        if len(acc_keys):
                            acc = logs_dict[acc_keys[-1]].mean()

                        if verbose_period is not None and idx_stimulus % verbose_period == 0:
                            descr = f"GPU{os.environ.get('CUDA_VISIBLE_DEVICES', None)}-Epoch {epoch+1} :: {mode} Iteration {idx_stimulus+1}/{len(data_loader)}"
                            if isinstance(loss, torch.Tensor): loss = loss.item()
                            descr += f" (Rep:{it_rep+1}/{nbr_experience_repetition}):: Loss {it+1} = {loss}"
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
                            self.stream_handler.update("signals:curriculum_nbr_distractors", nbr_distractors)
                            logger.add_scalar( "{}/CurriculumNbrDistractors".format(mode), nbr_distractors, it_step)
                            logger.add_scalar( "{}/CurriculumWindowedAcc".format(mode), windowed_accuracy, it_step)
                        
                        
                        # TODO: make this a logger module:
                        if 'current_speaker' in self.modules and 'current_listener' in self.modules:
                            prototype_speaker = self.stream_handler["modules:current_speaker:ref_agent"]
                            prototype_listener = self.stream_handler["modules:current_listener:ref_agent"]
                            image_save_path = logger.path 
                            if prototype_speaker is not None \
                            and hasattr(prototype_speaker,'VAE') \
                            and prototype_speaker.VAE.decoder is not None \
                            and idx_stimulus % 4 == 0:
                                query_vae_latent_space(prototype_speaker.VAE, 
                                                       sample=sample['speaker_experiences'],
                                                       path=image_save_path,
                                                       test=('test' in mode),
                                                       full=('test' in mode),
                                                       idxoffset=it_rep+idx_stimulus*self.config['nbr_experience_repetition'],
                                                       suffix='speaker',
                                                       use_cuda=True)
                                
                            if prototype_listener is not None \
                            and hasattr(prototype_listener,'VAE') \
                            and prototype_listener.VAE.decoder is not None \
                            and idx_stimulus % 4 == 0:
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
            if self.save_epoch_interval is not None\
             and epoch % self.save_epoch_interval == 0:
                self.save(path=self.save_path)

            # //------------------------------------------------------------//
            # //------------------------------------------------------------//
            # //------------------------------------------------------------//
        
        self.stream_handler.update("signals:epoch", init_epoch+nbr_epoch)

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------// 
        
        return



            




