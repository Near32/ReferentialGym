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
        if 'parallel_TS_computation_max_workers' not in self.config:
            self.config['parallel_TS_computation_max_workers'] = 16

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
        
        if self.config['use_curriculum_nbr_distractors']:
            windowed_accuracy = 0.0
            window_count = 0
            for mode in self.datasets:
                self.datasets[mode].setNbrDistractors(1)
            
        pbar = tqdm(total=nbr_epoch)
        
        for epoch in range(nbr_epoch):
            self.stream_handler.update("signals:epoch", epoch)
            pbar.update(1)
            for mode, data_loader in data_loaders.items():
                self.stream_handler.update("signals:mode", mode)
                counterGames = 0
                total_sentences = []
                total_nbr_unique_stimulus = 0
                epoch_acc = []

                for it_dataset in range(self.config['nbr_dataset_repetition'][mode]):
                    end_of_epoch_dataset = (it_dataset==self.config['nbr_dataset_repetition'][mode]-1)
                    self.stream_handler.update("signals:end_of_epoch_dataset", end_of_epoch_dataset)
                    
                    nbr_experience_repetition = self.config['nbr_experience_repetition'] if 'train' in mode else 1

                    for idx_stimuli, sample in enumerate(data_loader):
                        end_of_epoch_datasample = end_of_epoch_dataset and (idx_stimuli==len(data_loader)-1)
                        self.stream_handler.update("signals:end_of_epoch_datasample", end_of_epoch_datasample)
                        it_datasamples[mode] += 1
                        it_datasample = it_datasamples[mode]
                        self.stream_handler.update("signals:it_datasample", it_datasample)
                        it = it_datasample


                        if self.config['use_cuda']:
                            sample = sample.cuda()

                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        
                        for it_rep in range(nbr_experience_repetition):
                            it_samples[mode] += 1
                            it_sample = it_samples[mode]
                            end_of_epoch_sample = end_of_epoch_datasample and (it_rep==nbr_experience_repetition-1)
                            self.stream_handler.update("signals:end_of_epoch_sample", end_of_epoch_sample)
                            batch_size = len(sample['speaker_experiences'])
                            
                            # TODO: implement a multi_round_communicatioin module ?
                            for idx_round in range(self.config['nbr_communication_round']):
                                it_steps[mode] += 1
                                it_step = it_steps[mode]
                                self.stream_handler.update("signals:it_step", it_step)
                            
                                multi_round = True
                                if idx_round == self.config['nbr_communication_round']-1:
                                    multi_round = False
                                self.stream_handler.update("signals:multi_round", multi_round)
                                self.stream_handler.update('current_dataloader:sample', sample)

                                if "referential_game" in self.pipelines:
                                    self.stream_handler.serve(self.pipelines["referential_game"])

                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            
                            for pipe_id, pipeline in self.pipelines.items():
                                if pipe_id == "referential_game": continue
                                self.stream_handler.serve(pipeline)
                            
                            # Logging:        
                            logs_dict = self.stream_handler["logs_dict"]
                            for logname, value in logs_dict.items():
                                if isinstance(value, torch.Tensor): value = value.item()
                                logger.add_scalar(logname, value, it_step)    
                            self.stream_handler.reset("logs_dict")

                            #TODO: transform all of this below into modules:

                            losses = self.stream_handler["losses_dict"]
                            self.stream_handler.reset("losses_dict")

                            idx_loss = 1 if not self.config['use_homoscedastic_multitasks_loss'] else 2  
                            loss = sum( [l[idx_loss] for l in losses.values()])

                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            
                            # Compute sentences:
                            if 'current_speaker' in self.modules:
                                sentences = []
                                speaker_sentences_widx = self.stream_handler["modules:current_speaker:sentences_widx"]
                                for sidx in range(batch_size):
                                    #sentences.append(''.join([chr(97+int(s.item())) for s in speaker_outputs['sentences_widx'][sidx] ]))
                                    sentences.append(''.join([chr(97+int(s.item())) for s in speaker_sentences_widx[sidx] ]))
                                
                            if logger is not None:
                                if self.config['with_grad_logging'] and mode == 'train':
                                    maxgrad = 0.0
                                    for name, p in speaker.named_parameters() :
                                        if hasattr(p,'grad') and p.grad is not None:
                                            logger.add_histogram( "Speaker/{}".format(name), p.grad, idx_stimuli+len(data_loader)*epoch)
                                            cmg = torch.abs(p.grad).max()
                                            if cmg > maxgrad:
                                                maxgrad = cmg
                                    logger.add_scalar( "{}/SpeakerMaxGrad".format(mode), maxgrad, it_step)                    
                                    maxgrad = 0
                                    for name, p in listener.named_parameters() :
                                        if hasattr(p,'grad') and p.grad is not None:
                                            logger.add_histogram( "Listener/{}".format(name), p.grad, it_step)                    
                                            cmg = torch.abs(p.grad).max()
                                            if cmg > maxgrad:
                                                maxgrad = cmg
                                    logger.add_scalar( "{}/ListenerMaxGrad".format(mode), maxgrad, it_step)                    
                                
                                if self.config['with_utterance_penalization'] or self.config['with_utterance_promotion']:
                                    import ipdb; ipdb.set_trace()
                                    for widx in range(self.config['vocab_size']+1):
                                        logger.add_scalar("{}/Word{}Counts".format(mode,widx), speaker_outputs['speaker_utterances_count'][widx], it_step)
                                    logger.add_scalar("{}/OOVLoss".format(mode), speaker_losses['oov_loss'][idx_loss].mean().item(), it_step)

                                if 'with_mdl_principle' in self.config and self.config['with_mdl_principle']:
                                    logger.add_scalar("{}/MDLLoss".format(mode), speaker_losses['mdl_loss'][idx_loss].mean().item(), it_step)
                                
                                if 'current_speaker' in self.modules:
                                    if speaker_sentences_widx is not None:
                                        if 'obverter' in self.config['graphtype'].lower():
                                            sentence_length = torch.sum(-(speaker_sentences_widx.squeeze(-1)-self.config['vocab_size']).sign(), dim=-1).mean()
                                        else:
                                            sentence_length = (speaker_sentences_widx < (self.config['vocab_size']-1)).sum().float()/batch_size
                                        logger.add_scalar('{}/SentenceLength (/{})'.format(mode, self.config['max_sentence_length']), sentence_length/self.config['max_sentence_length'], it_step)
                                        
                                        for sentence in sentences:  total_sentences.append(sentence.replace(chr(97+self.config['vocab_size']), ''))
                                        total_nbr_unique_sentences = cardinality(total_sentences)
                                        total_nbr_unique_stimulus += batch_size
                                        logger.add_scalar('{}/Ambiguity (%)'.format(mode), float(total_nbr_unique_stimulus-total_nbr_unique_sentences)/total_nbr_unique_stimulus*100.0, it_step)
                                        
                                        speaker_sentences_logits = self.stream_handler["modules:current_speaker:sentences_logits"]
                                        speaker_sentences_widx = self.stream_handler["modules:current_speaker:sentences_widx"]
                                        sentences_log_probs = [s_logits.reshape(-1,self.config['vocab_size']).log_softmax(dim=-1) 
                                                                for s_logits in speaker_sentences_logits]
                                        speaker_sentences_log_probs = torch.cat(
                                            [s_log_probs.gather(dim=-1,index=s_widx[:s_log_probs.shape[0]].long()).sum().unsqueeze(0) 
                                            for s_log_probs, s_widx in zip(sentences_log_probs, speaker_sentences_widx)], 
                                            dim=0)
                                        entropies_per_sentence = -(speaker_sentences_log_probs.exp() * speaker_sentences_log_probs)
                                        # (batch_size, )
                                        logger.add_scalar('{}/Entropy'.format(mode), entropies_per_sentence.mean().item(), it_step)
                                
                                    speaker_maxl1_loss = self.stream_handler["modules:current_speaker:maxl1_loss"]
                                    listener_maxl1_loss = self.stream_handler["modules:current_listener:maxl1_loss"]
                                    if speaker_maxl1_loss is not None and listener_maxl1_loss is not None:
                                        logger.add_scalar('{}/WeightMaxL1Loss'.format(mode), speaker_maxl1_loss.item()+listener_maxl1_loss.item(), it_step)
                                    
                                
                                    # Compute ACCURACIES:
                                    #decision_probs = listener_outputs['decision_probs']
                                    decision_probs = self.stream_handler["modules:current_listener:decision_probs"]
                                    # TODO: make it a module rather?
                                    if decision_probs is not None:
                                        try:
                                            if self.config['descriptive']:
                                                import ipdb; ipdb.set_trace()
                                                if 'obverter_least_effort_loss' in self.config and self.config['obverter_least_effort_loss']:
                                                    for widx in range(len(losses4widx)):
                                                        logger.add_scalar('{}/Loss@w{}'.format(mode,widx), losses4widx[widx].item(), it_step)
                                                        acc = (torch.abs(decision_probs[:,widx,...]-target_decision_probs) < 0.5).float().mean()*100
                                                        logger.add_scalar('{}/Accuracy@{}'.format(mode,widx), acc.item(), it_step)
                                                else:
                                                    #acc = (torch.abs(decision_probs-target_decision_probs) < 0.5).float().mean()*100
                                                    decision_idx = decision_probs.max(dim=-1)[1]
                                                    acc = (decision_idx==sample['target_decision_idx']).float()#.mean()*100
                                            else:
                                                decision_idx = decision_probs.max(dim=-1)[1]
                                                acc = (decision_idx==sample['target_decision_idx']).float()#.mean()*100
                                            
                                            logger.add_scalar('{}/Accuracy'.format(mode), acc.mean().item()*100, it_step)
                                            epoch_acc.append(acc.view(-1))
                                            if end_of_epoch_sample:
                                                epoch_acc = torch.cat(epoch_acc, dim=0).mean()
                                                logger.add_scalar('{}/PerEpoch/Accuracy'.format(mode), epoch_acc.item()*100, epoch)
                                                epoch_acc = []
                                        except Exception as e:
                                            print(f"Erreur in computing accuracies... {e}")
                                            import ipdb; ipdb.set_trace()

                                logger.add_scalar( "{}/Stimulus/Mean".format(mode), sample['listener_experiences'].mean(), it_step)
                                
                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            # //------------------------------------------------------------//
                            

                            if verbose_period is not None and idx_stimuli % verbose_period == 0:
                                descr = 'Epoch {} :: {} Iteration {}/{} :: Loss {} = {}'.format(epoch, mode, idx_stimuli, len(data_loader), it, loss.item())
                                pbar.set_description_str(descr)
                                # Show some sentences:
                                if VERBOSE:
                                    for sidx in range(batch_size):
                                        if sidx>=10:
                                            break
                                        print(f"{sidx} : ", sentences[sidx], \
                                            f"\t /label: {sample['target_decision_idx'][sidx]}",\
                                            f" /decision: {decision_idx[sidx]}")

                            # TODO: CURRICULUM ON DISTRATORS as a module that handles the current dataloader reference....!!
                            if self.config['use_curriculum_nbr_distractors']:
                                nbr_distractors = self.datasets[mode].getNbrDistractors()
                                logger.add_scalar( "{}/CurriculumNbrDistractors".format(mode), nbr_distractors, it_step)
                                logger.add_scalar( "{}/CurriculumWindowedAcc".format(mode), windowed_accuracy, it_step)
                            
                            
                            # TODO: make this a logger module:
                            if 'current_speaker' in self.modules and 'current_listener' in self.modules:
                                prototype_speaker = self.stream_handler["modules:current_speaker:ref_agent"]
                                prototype_listener = self.stream_handler["modules:current_listener:ref_agent"]
                                if prototype_speaker is not None and hasattr(prototype_speaker,'VAE') and idx_stimuli % 4 == 0:
                                    import ipdb; ipdb.set_trace()
                                    image_save_path = logger.path 
                                    '''
                                    '''
                                    query_vae_latent_space(prototype_speaker.VAE, 
                                                           sample=sample['speaker_experiences'],
                                                           path=image_save_path,
                                                           test=('test' in mode),
                                                           full=('test' in mode),
                                                           idxoffset=it_rep+idx_stimuli*self.config['nbr_experience_repetition'],
                                                           suffix='speaker',
                                                           use_cuda=True)
                                    query_vae_latent_space(prototype_listener.VAE, 
                                                           sample=sample['listener_experiences'],
                                                           path=image_save_path,
                                                           test=('test' in mode),
                                                           full=('test' in mode),
                                                           idxoffset=idx_stimuli,
                                                           suffix='listener')
                                    
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//
                        # //------------------------------------------------------------//

                        # TODO: many parts everywhere, do not forget them all : CURRICULUM ON DISTRACTORS...!!!
                        if self.config['use_curriculum_nbr_distractors'] and 'train' in mode:
                            nbr_distractors = self.datasets[mode].getNbrDistractors()
                            windowed_accuracy = (windowed_accuracy*window_count+acc.item())
                            window_count += 1
                            windowed_accuracy /= window_count
                            if windowed_accuracy > 60 and window_count > self.config['curriculum_distractors_window_size'] and nbr_distractors < self.config['nbr_distractors'] :
                                windowed_accuracy = 0
                                window_count = 0
                                for mode in self.dataset:
                                    self.datasets[mode].setNbrDistractors(self.datasets[mode].getNbrDistractors()+1)
                        # //------------------------------------------------------------//
                
                # COMPUTE TOPOGRAPHIC SIMILARITIES:
                # TODO: make it a logging module...
                #       The history of the data stream in the current epoch should be registered in a growing list/stream?
                if logger is not None and 'current_speaker' in self.modules:
                    if 'test' in mode:  
                        max_nbr_samples = None
                    else:   
                        max_nbr_samples = int(len(total_sentences)*0.1)

                    try:
                        topo_sims, pvalues, unique_prod_ratios = logger.measure_topographic_similarity(sentences_key='sentences_widx',
                                                                                   features_key='exp_latents',
                                                                                   max_nbr_samples=max_nbr_samples,
                                                                                   verbose=VERBOSE,
                                                                                   max_workers=self.config['parallel_TS_computation_max_workers'])
                        topo_sims_v, pvalues_v, unique_prod_ratios_v = logger.measure_topographic_similarity(sentences_key='sentences_widx',
                                                                                   features_key='exp_latents_values',
                                                                                   max_nbr_samples=max_nbr_samples,
                                                                                   verbose=VERBOSE,
                                                                                   max_workers=self.config['parallel_TS_computation_max_workers'])
                        feat_topo_sims, feat_pvalues, _ = logger.measure_topographic_similarity(sentences_key='sentences_widx',
                                                                                   features_key='temporal_features',
                                                                                   max_nbr_samples=max_nbr_samples,
                                                                                   verbose=VERBOSE,
                                                                                   max_workers=self.config['parallel_TS_computation_max_workers'])
                        
                        for agent_id in topo_sims:
                            logger.add_scalar('{}/{}/TopographicSimilarity'.format(mode,agent_id), topo_sims[agent_id], epoch)
                            logger.add_scalar('{}/{}/TopographicSimilarity-UniqueProduction'.format(mode,agent_id), unique_prod_ratios[agent_id], epoch)
                            logger.add_scalar('{}/{}/TopographicSimilarity-PValues'.format(mode,agent_id), pvalues[agent_id], epoch)
                        for agent_id in topo_sims_v:
                            logger.add_scalar('{}/{}/TopographicSimilarity_withValues'.format(mode,agent_id), topo_sims_v[agent_id], epoch)
                            logger.add_scalar('{}/{}/TopographicSimilarity_withValues-PValues'.format(mode,agent_id), pvalues_v[agent_id], epoch)
                        for agent_id in feat_topo_sims:
                            logger.add_scalar('{}/{}/FeaturesTopographicSimilarity (%)'.format(mode,agent_id), feat_topo_sims[agent_id], epoch)
                    except Exception as e:
                        print(f"Erreur in computing topo sims... {e}")
                        import ipdb; ipdb.set_trace()

                
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



            




