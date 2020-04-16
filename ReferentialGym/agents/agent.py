from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np 
import random 

import copy

from ..modules import Module

def havrylov_hinge_learning_signal(decision_logits, target_decision_idx, sampled_decision_idx=None, multi_round=False):
    target_decision_logits = decision_logits.gather(dim=1, index=target_decision_idx)
    # (batch_size, 1)

    distractors_logits_list = [torch.cat([pb_dl[:tidx.item()], pb_dl[tidx.item()+1:]], dim=0).unsqueeze(0) 
        for pb_dl, tidx in zip(decision_logits, target_decision_idx)]
    distractors_decision_logits = torch.cat(
        distractors_logits_list, 
        dim=0)
    # (batch_size, nbr_distractors)
    
    loss_element = 1-target_decision_logits+distractors_decision_logits
    # (batch_size, nbr_distractors)
    maxloss_element = torch.max(torch.zeros_like(loss_element), loss_element)
    loss = maxloss_element.sum(dim=1)
    # (batch_size, )

    done = (target_decision_idx == decision_logits.argmax(dim=-1))
    
    return loss, done


def penalize_multi_round_binary_reward_fn(sampled_decision_idx, target_decision_idx, decision_logits=None, multi_round=False):
    '''
    Computes the reward and done boolean of the current timestep.
    Episode ends if the decisions are correct 
    (or if the max number of round is achieved, but this is handled outside of this function).
    '''
    done = (sampled_decision_idx == target_decision_idx)
    r = done.float()*torch.ones_like(target_decision_idx)
    if multi_round:
        r -= 0.1
    return -r, done


class ExperienceBuffer(object):
    def __init__(self, capacity, keys=None, circular_keys={'succ_s':'s'}, circular_offsets={'succ_s':1}):
        '''
        Use a different circular offset['succ_s']=n to implement truncated n-step return...
        '''
        if keys is None:    keys = ['s', 'a', 'r', 'non_terminal', 'rnn_state']
        self.keys = keys
        self.circular_keys = circular_keys
        self.circular_offsets = circular_offsets
        self.capacity = capacity
        self.position = dict()
        self.current_size = dict()
        self.reset()

    def add_key(self, key):
        self.keys += [key]
        #setattr(self, key, np.zeros(self.capacity+1, dtype=object))
        setattr(self, key, [None]*(self.capacity+1))
        self.position[key] = 0
        self.current_size[key] = 0

    def add(self, data):
        for k, v in data.items():
            if not(k in self.keys or k in self.circular_keys):  continue
            if k in self.circular_keys: continue
            getattr(self, k)[self.position[k]] = v
            self.position[k] = int((self.position[k]+1) % self.capacity)
            self.current_size[k] = min(self.capacity, self.current_size[k]+1)

    def pop(self):
        '''
        Output a data dict of the latest 'complete' data experience.
        '''
        all_keys = self.keys+list(self.circular_keys.keys())
        max_offset = 0
        if len(self.circular_offsets):
            max_offset = max([offset for offset in self.circular_offsets.values()])
        data = {k:None for k in self.keys}
        for k in all_keys:
            fetch_k = k
            offset = 0
            if k in self.circular_keys: 
                fetch_k = self.circular_keys[k]
                offset = self.circular_offsets[k]
            next_position_write = self.position[fetch_k] 
            position_complete_read_possible = (next_position_write-1)-max_offset
            k_read_position = position_complete_read_possible+offset 
            data[k] = getattr(self, fetch_k)[k_read_position]
        return data 

    def reset(self):
        for k in self.keys:
            if k in self.circular_keys: continue
            #setattr(self, k, np.zeros(self.capacity+1, dtype=object))
            setattr(self, k, [None]*(self.capacity+1))
            self.position[k] = 0
            self.current_size[k] = 0

    def cat(self, keys, indices=None):
        data = []
        for k in keys:
            assert k in self.keys or k in self.circular_keys, f'Tried to get value from key {k}, but {k} is not registered.'
            indices_ = indices
            cidx=0
            if k in self.circular_keys: 
                cidx=self.circular_offsets[k]
                k = self.circular_keys[k]
            v = getattr(self, k)
            if indices_ is None: indices_ = np.arange(self.current_size[k]-1-cidx)
            else:
                # Check that all indices are in range:
                for idx in range(len(indices_)):
                    if self.current_size[k]>0 and indices_[idx]>=self.current_size[k]-1-cidx:
                        indices_[idx] = np.random.randint(self.current_size[k]-1-cidx)
                        # propagate to argument:
                        indices[idx] = indices_[idx]
            '''
            '''
            indices_ = cidx+indices_
            values = v[indices_]
            data.append(values)
        return data 

    def __len__(self):
        return self.current_size[self.keys[0]]

    def sample(self, batch_size, keys=None):
        if keys is None:    keys = self.keys + self.circular_keys.keys()
        min_current_size = self.capacity
        for idx_key in reversed(range(len(keys))):
            key = keys[idx_key]
            if key in self.circular_keys:   key = self.circular_keys[key]
            if self.current_size[key] == 0:
                continue
            if self.current_size[key] < min_current_size:
                min_current_size = self.current_size[key]

        indices = np.random.choice(np.arange(min_current_size-1), batch_size)
        data = self.cat(keys=keys, indices=indices)
        return data


class Agent(Module):
    def __init__(self, 
                 agent_id='l0', 
                 obs_shape=[1,1,1,32,32], 
                 vocab_size=100, 
                 max_sentence_length=10, 
                 logger=None, 
                 kwargs=None,
                 role=None):
        """
        :param agent_id: str defining the ID of the agent over the population.
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        :param role: str defining the role of the agent, e.g. "speaker"/"listener".
        """

        input_stream_keys = {'speaker':list(), 'listener':list()}
        input_stream_ids = {'speaker':list(), 'listener':list()}
        
        input_stream_ids['speaker'] = {
            'current_dataloader:sample:speaker_experiences':'experiences', 
            'current_dataloader:sample:speaker_exp_latents':'exp_latents', 
            'current_dataloader:sample:speaker_exp_latents_values':'exp_latents_values', 
            'modules:current_listener:sentences_logits':'sentences_logits',
            'modules:current_listener:sentences_one_hot':'sentences_one_hot',
            'modules:current_listener:sentences_widx':'sentences_widx', 
            'config':'config',
            'config:graphtype':'graphtype',
            'config:tau0':'tau0',
            'signals:multi_round':'multi_round',
            'signals:end_of_epoch_sample':'end_of_epoch_sample',
            'signals:mode':'mode',
            'signals:it_step':'it',
            'current_dataloader:sample':'sample',
            'losses_dict':'losses_dict',
            'logs_dict':'logs_dict',
        }

        input_stream_ids['listener'] = {
            'current_dataloader:sample:listener_experiences':'experiences', 
            'current_dataloader:sample:listener_exp_latents':'exp_latents', 
            'current_dataloader:sample:listener_exp_latents_values':'exp_latents_values', 
            'modules:current_speaker:sentences_logits':'sentences_logits',
            'modules:current_speaker:sentences_one_hot':'sentences_one_hot',
            'modules:current_speaker:sentences_widx':'sentences_widx', 
            'config':'config',
            'config:graphtype':'graphtype',
            'config:tau0':'tau0',
            'signals:multi_round':'multi_round',
            'signals:end_of_epoch_sample':'end_of_epoch_sample',
            'signals:mode':'mode',
            'signals:it_step':'it',
            'current_dataloader:sample':'sample',
            'losses_dict':'losses_dict',
            'logs_dict':'logs_dict',
        }

        super(Agent, self).__init__(id=agent_id,
                                    type="Agent", 
                                    config=kwargs,
                                    input_stream_ids=input_stream_ids)
        
        self.agent_id = agent_id
        
        self.obs_shape = obs_shape
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        
        self.logger = logger 
        self.kwargs = kwargs

        self.log_idx = 0
        self.log_dict = dict()

        self.use_sentences_one_hot_vectors = False

        self.vocab_start_idx = 0
        self.vocab_stop_idx = 1
        self.vocab_pad_idx = self.vocab_size-1       
        
        self.hooks = []

        # REINFORCE algorithm:
        self.gamma = 0.99
        #self.learning_signal_pred_fn = penalize_multi_round_binary_loss_fn
        self.learning_signal_pred_fn = havrylov_hinge_learning_signal
        self.exp_buffer = ExperienceBuffer(capacity=self.kwargs['nbr_communication_round']*2,
                                           keys=['speaker_sentences_entrs',
                                                 'speaker_sentences_token_bool',
                                                 'speaker_sentences_log_probs',
                                                 'listener_sentences_entrs',
                                                 'listener_sentences_log_probs',
                                                 'decision_entrs',
                                                 'decision_log_probs',
                                                 'r',
                                                 'done'],
                                           circular_keys={}, 
                                           circular_offsets={})

        self.learning_signal_baseline = 0.0
        self.learning_signal_baseline_counter = 0

        self.role = role        
    
    def get_input_stream_keys(self):
        return self.input_stream_keys[self.role]

    def get_input_stream_ids(self):
        return self.input_stream_ids[self.role]

    def clone(self, clone_id='a0'):
        logger = self.logger
        self.logger = None 
        clone = copy.deepcopy(self)
        clone.agent_id = clone_id
        clone.logger = logger 
        self.logger = logger  
        return clone 

    def save(self, path):
        logger = self.logger
        self.logger = None
        torch.save(self, path)
        self.logger = logger 

    def _tidyup(self):
        pass 
    
    def _log(self, log_dict, batch_size):
        if self.logger is None: 
            return 

        agent_log_dict = {f'{self.agent_id}': dict()}
        for key, data in log_dict.items():
            if data is None:
                data = [None]*batch_size
            agent_log_dict[f'{self.agent_id}'].update({f'{key}':data})
        
        self.logger.add_dict(agent_log_dict, batch=True, idx=self.log_idx) 
        
        self.log_idx += 1

    def register_hook(self, hook):
        self.hooks.append(hook)

    def forward(self, sentences, experiences, multi_round=False, graphtype='straight_through_gumbel_softmax', tau0=0.2):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        raise NotImplementedError

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Compute the losses and return them along with the produced outputs.

        :param input_streams_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_logits'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits over symbols.
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sentences_one_hot'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
            - `'exp_latents'`: Tensor of shape `(batch_size, nbr_latent_dimensions)`.
            - `'multi_round'`: Boolean defining whether to utter a sentence back or not.
            - `'graphtype'`: String defining the type of symbols used in the output sentence:
                        - `'categorical'`: one-hot-encoded symbols.
                        - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                        - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                        - `'obverter'`: obverter training scheme...
            - `'tau0'`: Float, temperature with which to apply gumbel-softmax estimator. 
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
            - `'config'`: Dict of hyperparameters to the referential game.
            - `'mode'`: String that defines what mode we are in, e.g. 'train' or 'test'. Those keywords are expected.
            - `'it'`: Integer specifying the iteration number of the current function call.
        """
        config = input_streams_dict['config']
        mode = input_streams_dict['mode']
        it = input_streams_dict['it']
        losses_dict = input_streams_dict['losses_dict']
        logs_dict = input_streams_dict['logs_dict']
        
        batch_size = len(input_streams_dict['experiences'])

        input_sentence = input_streams_dict['sentences_widx']
        if self.use_sentences_one_hot_vectors:
            input_sentence = input_streams_dict['sentences_one_hot']

        outputs_dict = self(sentences=input_sentence,
                           experiences=input_streams_dict['experiences'],
                           multi_round=input_streams_dict['multi_round'],
                           graphtype=input_streams_dict['graphtype'],
                           tau0=input_streams_dict['tau0'])

        outputs_dict['exp_latents'] = input_streams_dict['exp_latents']
        outputs_dict['exp_latents_values'] = input_streams_dict['exp_latents_values']
        self._log(outputs_dict, batch_size=batch_size)

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//

        weight_maxl1_loss = 0.0
        for p in self.parameters() :
            weight_maxl1_loss += torch.max( torch.abs(p) )
        outputs_dict['maxl1_loss'] = weight_maxl1_loss        

        for hook in self.hooks:
            hook(losses_dict=losses_dict,
                    log_dict=self.log_dict,
                    inputs_dict=input_streams_dict,
                    outputs_dict=outputs_dict,
                    mode=mode,
                    role=self.role
                    )

        if hasattr(self,'tau'): 
            tau = torch.cat([ t.view((-1)) for t in self.tau], dim=0) if isinstance(self.tau, list) else self.tau
            logs_dict[f"{mode}/Tau/{self.role}/Mean"] = tau.mean().item()
            logs_dict[f"{mode}/Tau/{self.role}/Std"] = tau.std().item()

        
        '''
        if hasattr(self, 'TC_losses'):
            losses_dict[f'{self.role}/TC_loss'] = [1.0, self.TC_losses]
        '''
        if hasattr(self, 'VAE_losses'):# and ('listener' in role or not('obverter' in input_streams_dict['graphtype'])):
            losses_dict[f'{self.role}/VAE_loss'] = [self.kwargs['VAE_lambda'], self.VAE_losses]

        if 'speaker' in self.role:
            if ('with_utterance_penalization' in config or 'with_utterance_promotion' in config) and (config['with_utterance_penalization'] or config['with_utterance_promotion']):
                arange_vocab = torch.arange(config['vocab_size']+1).float()
                if config['use_cuda']: arange_vocab = arange_vocab.cuda()
                speaker_utterances = torch.cat( \
                    [((s+1) / (s.detach()+1)) * torch.nn.functional.one_hot(s.long().squeeze(), num_classes=config['vocab_size']+1).float().unsqueeze(0) \
                    for s in outputs_dict['sentences_widx']], 
                    dim=0)
                # (batch_size, sentence_length,vocab_size+1)
                speaker_utterances_count = speaker_utterances.sum(dim=0).sum(dim=0).float().squeeze()
                outputs_dict['speaker_utterances_count'] = speaker_utterances_count
                # (vocab_size+1,)
                total_nbr_utterances = speaker_utterances_count.sum().item()
                d_speaker_utterances_probs = (speaker_utterances_count/(config['utterance_oov_prob']+total_nbr_utterances-1)).detach()
                # (vocab_size+1,)
                #oov_loss = -(1.0/(batch_size*config['max_sentence_length']))*torch.sum(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
                oov_loss = -(1.0/(batch_size*config['max_sentence_length']))*(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
                # (batch_size, 1)
                if config['with_utterance_promotion']:
                    oov_loss *= -1 
                losses_dict['oov_loss'] = [config['utterance_factor'], oov_loss]

            if 'with_mdl_principle' in config and config['with_mdl_principle']:
                '''
                speaker_utterances_max_logit = [torch.max(s, dim=-1)[0].view((1,-1)) for s in outputs_dict['sentences_logits']]
                # (batch_size, (1,sentence_length))
                mask = [ (l < config['obverter_stop_threshold']).float() for l in speaker_utterances_max_logit]
                # (batch_size, (1,sentence_lenght))
                if config['use_cuda']: mask = [ m.cuda() for m in mask]
                # (batch_size, (1,sentence_lenght))
                #mdl_loss = -(mask*speaker_utterances_max_logit).sum()/(batch_size*self.config['max_sentence_length'])
                mdl_loss = torch.cat([ (m*(1-l)).sum(dim=-1)/(batch_size) for m,l in zip(mask,speaker_utterances_max_logit)], dim=0)
                # (batch_size, )
                losses_dict['mdl_loss'] = [config['mdl_principle_factor'], mdl_loss]
                '''
                arange_token = torch.arange(config['max_sentence_length'])
                arange_token = (config['vocab_size']*arange_token).float().view((1,-1)).repeat(batch_size,1)
                if config['use_cuda']: arange_token = arange_token.cuda()
                mask = (outputs_dict['sentences_widx'] < (config['vocab_size']-1)).float()
                # (batch_size, max_sentence_length, 1)
                if config['use_cuda']: mask = mask.cuda()
                speaker_reweighted_utterances = 1+mask*outputs_dict['sentences_widx']+(-1)*(1-mask)*outputs_dict['sentences_widx']/(config['vocab_size']-1)
                mdl_loss = (arange_token+speaker_reweighted_utterances.squeeze()).mean(dim=-1)
                # (batch_size, )
                losses_dict['mdl_loss'] = [config['mdl_principle_factor'], mdl_loss]
                
            if 'with_speaker_entropy_regularization' in config and config['with_speaker_entropy_regularization']:
                entropies_per_sentence = torch.cat([torch.cat([ torch.distributions.categorical.Categorical(logits=w_logits).entropy().view(1,1) for w_logits in s_logits], dim=-1).mean(dim=-1) for s_logits in outputs_dict['sentences_logits']], dim=0)
                # (batch_size, 1)
                losses_dict['speaker_entropy_regularization_loss'] = [config['entropy_regularization_factor'], entropies_per_sentence.squeeze()]
                # (batch_size, )
            if config['with_weight_maxl1_loss']:
                losses_dict['speaker_maxl1_weight_loss'] = [1.0, weight_maxl1_loss]


        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        

        if 'listener' in self.role:
            sample = input_streams_dict['sample']
            
            decision_logits = outputs_dict['decision']
            final_decision_logits = decision_logits
            # (batch_size, max_sentence_length / squeezed if not using obverter agent, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            if 'straight_through_gumbel_softmax' in config['graphtype'].lower() or 'obverter' in config['graphtype'].lower():
                if config['agent_loss_type'].lower() == 'nll':
                    if config['descriptive']:  
                        decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                        criterion = nn.NLLLoss(reduction='none')
                        
                        if 'obverter_least_effort_loss' in config and config['obverter_least_effort_loss']:
                            loss = 0.0
                            losses4widx = []
                            for widx in range(decision_probs.size(1)):
                                dp = decision_probs[:,widx,...]
                                ls = criterion( dp, sample['target_decision_idx'])
                                loss += config['obverter_least_effort_loss_weights'][widx]*ls 
                                losses4widx.append(ls)
                        else:
                            decision_probs = decision_probs[:,-1,...]
                            loss = criterion( decision_probs, sample['target_decision_idx'])
                            # (batch_size, )
                    else:   
                        final_decision_logits = final_decision_logits[:,-1,...]
                        # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
                        decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                        criterion = nn.NLLLoss(reduction='none')
                        loss = criterion( final_decision_logits, sample['target_decision_idx'])
                        # (batch_size, )
                    losses_dict['referential_game_loss'] = [1.0, loss]
                elif config['agent_loss_type'].lower() == 'hinge':
                    #Havrylov's Hinge loss:
                    if 'obverter' in config['graphtype'].lower():
                        sentences_lengths = torch.sum(-(input_streams_dict['sentences_widx'].squeeze(-1)-self.vocab_size).sign(), dim=-1).long()
                        # (batch_size,) 
                        sentences_lengths = sentences_lengths.reshape(
                            -1,
                            1,
                            1
                        ).expand(
                            final_decision_logits.shape[0],
                            1,
                            final_decision_logits.shape[2]
                        )
                        # (batch_size, max_sentence_length, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent)) 
                        final_decision_logits = final_decision_logits.gather(dim=1, index=(sentences_lengths-1)).squeeze(1)
                    else:
                        final_decision_logits = final_decision_logits[:,-1]
                    # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
                    decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                    
                    loss, _ = havrylov_hinge_learning_signal(decision_logits=final_decision_logits,
                                                          target_decision_idx=sample['target_decision_idx'].unsqueeze(1),
                                                          multi_round=input_streams_dict['multi_round'])
                    # (batch_size, )
                    
                    losses_dict['referential_game_loss'] = [1.0, loss]    
                    '''
                    # Entropy minimization:
                    distr = torch.distributions.Categorical(probs=decision_probs)
                    entropy_loss = distr.entropy()
                    losses_dict['entropy_loss'] = [1.0, entropy_loss]
                    '''
                else:
                    raise NotImplementedError

            elif 'reinforce' in config['graphtype'].lower():
                #REINFORCE Policy-gradient algorithm:

                #Compute data:
                final_decision_logits = final_decision_logits[:,-1,...]
                # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
                decision_probs = final_decision_logits.softmax(dim=-1)
                decision_distrs = Categorical(probs=decision_probs) 
                decision_entrs = decision_distrs.entropy()
                if 'argmax' in config['graphtype'].lower() and not(self.training):
                    sampled_decision_idx = final_decision_logits.argmax(dim=-1).unsqueeze(-1)
                else:
                    sampled_decision_idx = decision_distrs.sample().unsqueeze(-1)
                # (batch_size, 1)
                # It is very important to squeeze the sampled indices vector before computing log_prob,
                # otherwise the shape is broadcasted...
                decision_log_probs = decision_distrs.log_prob(sampled_decision_idx.squeeze()).unsqueeze(-1)
                # (batch_size, 1)
                
                target_decision_idx = sample['target_decision_idx'].unsqueeze(1)
                # (batch_size, 1)
                
                learning_signal, done = self.learning_signal_pred_fn(sampled_decision_idx=sampled_decision_idx,
                                                       decision_logits=final_decision_logits,
                                                       target_decision_idx=target_decision_idx,
                                                       multi_round=input_streams_dict['multi_round'])
                r = -learning_signal

                # Where are the actual token (different from padding):
                speaker_sentences_token_bool = (input_streams_dict['sentences_widx'] != self.vocab_pad_idx)
                # (batch_size, max_sentence_length, 1)
                
                # Compute sentences log_probs:
                speaker_sentences_logits = input_streams_dict['sentences_logits']
                # (batch_size, max_sentence_length, vocab_size)
                pad_token_logit = torch.zeros_like(speaker_sentences_logits[0][0]).unsqueeze(0)
                pad_token_logit[:, self.vocab_pad_idx] = 1.0
                # (1, vocab_size,)

                for b in range(len(speaker_sentences_logits)):
                    remainder = self.kwargs['max_sentence_length'] - len(speaker_sentences_logits[b])
                    if remainder > 0:
                        speaker_sentences_logits[b] = torch.cat([speaker_sentences_logits[b], pad_token_logit.repeat(remainder,1)], dim=0)
                    speaker_sentences_logits[b] = speaker_sentences_logits[b].unsqueeze(0)
                speaker_sentences_logits = torch.cat(speaker_sentences_logits, dim=0)

                speaker_sentences_log_probs = F.log_softmax(speaker_sentences_logits, dim=-1)
                # (batch_size, max_sentence_length, vocab_size)
                speaker_sentences_probs = speaker_sentences_log_probs.softmax(dim=-1)
                # (batch_size, max_sentence_length, vocab_size)
                speaker_sentences_log_probs = speaker_sentences_log_probs.gather(dim=-1, 
                    index=input_streams_dict['sentences_widx'].long())
                # (batch_size, max_sentence_length, 1)
                speaker_sentences_probs = speaker_sentences_probs.gather(dim=-1, 
                    index=input_streams_dict['sentences_widx'].long())
                # (batch_size, max_sentence_length, 1)
                
                # Entropy:
                speaker_sentences_entrs = -(speaker_sentences_probs * speaker_sentences_log_probs)
                # (batch_size, max_sentence_length, 1)
                
                listener_sentences_log_probs = None
                listener_sentences_entrs = None
                listener_sentences_token_bool = None 
                if input_streams_dict['multi_round']:  
                    # Where are the actual token (different from padding):
                    listener_sentences_token_bool = (outputs_dict['sentences_widx'] != self.vocab_pad_idx)
                    # (batch_size, max_sentence_length, 1)
                    
                    listener_sentences_logits = outputs_dict['sentences_logits']
                    # (batch_size, max_sentence_length, vocab_size)
                    for b in range(len(listener_sentences_logits)):
                        remainder = self.kwargs['max_sentence_length'] - len(listener_sentences_logits[b])
                        if remainder > 0:
                            listener_sentences_logits[b] = torch.cat([listener_sentences_logits[b], pad_token_logit.repeat(remainder,1)], dim=0)
                        listener_sentences_logits[b] = listener_sentences_logits[b].unsqueeze(0)
                    listener_sentences_logits = torch.cat(listener_sentences_logits, dim=0)
                    listener_sentences_log_probs = F.log_softmax(listener_sentences_logits, dim=-1)
                    # (batch_size, max_sentence_length, vocab_size)
                    listener_sentences_probs = listener_sentences_log_probs.softmax(dim=-11)
                    # (batch_size, max_sentence_length, vocab_size)
                    listener_sentences_log_probs = listener_sentences_log_probs.gather(dim=-1, 
                        index=outputs_dict['sentences_widx'].long())
                    # (batch_size, max_sentence_length, 1)
                    listener_sentences_probs = listener_sentences_probs.gather(dim=-1, 
                        index=input_streams_dict['sentences_widx'].long())
                    # (batch_size, max_sentence_length, 1)
                    
                    # Entropy:
                    listener_sentences_entrs = -(listener_sentences_probs * listener_sentences_log_probs)
                    # (batch_size, max_sentence_length, 1)
                        
                # Record data:
                data = {'speaker_sentences_entrs':speaker_sentences_entrs,
                        'speaker_sentences_token_bool':speaker_sentences_token_bool,
                        'speaker_sentences_log_probs':speaker_sentences_log_probs,
                        'listener_sentences_entrs':listener_sentences_entrs,
                        'listener_sentences_token_bool':listener_sentences_token_bool,
                        'listener_sentences_log_probs':listener_sentences_log_probs,
                        'decision_entrs':decision_entrs,
                        'decision_log_probs':decision_log_probs,
                        'r':r,
                        'done':done}

                self.exp_buffer.add(data)

                # Compute losses:
                if not(input_streams_dict['multi_round']):
                    # then it is the last round, we can compute the loss:
                    exp_size = len(self.exp_buffer)
                    R = torch.zeros_like(r)
                    returns = []
                    for r in reversed(self.exp_buffer.r[:exp_size]):
                        R = r + self.gamma * R
                        returns.insert(0, R.view(-1,1))
                        # (batch_size, 1)
                    returns = torch.cat(returns, dim=-1)
                    # (batch_size, nbr_communication_round)
                    # Normalized:
                    normalized_learning_signal = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-8)
                    # Unnormalized:
                    learning_signal = returns #(returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-8)
                    # (batch_size, nbr_communication_round)
                    
                    ls = learning_signal
                    if 'normalized' in config['graphtype'].lower():
                        ls = normalized_learning_signal

                    for it_round in range(learning_signal.shape[1]):
                        self.log_dict[f'{self.agent_id}/learning_signal_{it_round}'] = learning_signal[:,it_round].mean()
                    
                    formatted_baseline = 0.0
                    if 'baseline_reduced' in config['graphtype'].lower():
                        if self.training:
                            self.learning_signal_baseline = (self.learning_signal_baseline*self.learning_signal_baseline_counter+ls.detach().mean(dim=0))/(self.learning_signal_baseline_counter+1)
                            self.learning_signal_baseline_counter += 1
                        formatted_baseline = self.learning_signal_baseline.reshape(1,-1).repeat(batch_size,1).to(learning_signal.device)
                    
                        for it_round in range(learning_signal.shape[1]):
                            self.log_dict[f'{self.agent_id}/learning_signal_baseline_{it_round}'] = self.learning_signal_baseline[it_round].mean()    
                    
                    # Deterministic:
                    listener_decision_loss_deterministic = -learning_signal.sum(dim=-1)
                    # (batch_size, )
                    listener_decision_loss = listener_decision_loss_deterministic
                    
                    if 'stochastic' in config['graphtype'].lower():
                        # Stochastic:
                        decision_log_probs = torch.cat(self.exp_buffer.decision_log_probs[:exp_size], dim=-1)
                        # (batch_size, nbr_communication_round)
                        listener_decision_loss_stochastic = -(decision_log_probs * (ls.detach()-formatted_baseline)).sum(dim=-1)
                        # (batch_size, )
                        listener_decision_loss = listener_decision_loss_deterministic+listener_decision_loss_stochastic
                        
                    losses_dict['referential_game_loss'] = [1.0, listener_decision_loss]    
                    
                    # Decision Entropy loss:
                    decision_entropy = torch.cat(self.exp_buffer.decision_entrs[:exp_size], dim=-1).mean(dim=-1)
                    # (batch_size, )
                    self.log_dict[f'{self.agent_id}/decision_entropy'] = decision_entropy.mean()
                    #losses_dict['decision_entropy'] = [1.0, speaker_entropy_loss]    
                    

                    speaker_sentences_log_probs = torch.cat(self.exp_buffer.speaker_sentences_log_probs[:exp_size], dim=-1)
                    # (batch_size, max_sentence_length, nbr_communication_round)
                    # The log likelihood of each sentences is the sum (in log space) over each next token prediction:
                    speaker_sentences_log_probs = speaker_sentences_log_probs.sum(1)
                    # (batch_size, nbr_communication_round)
                    speaker_policy_loss = -(speaker_sentences_log_probs * (ls.detach()-formatted_baseline)).sum(dim=-1)
                    # (batch_size, )
                    losses_dict['speaker_policy_loss'] = [-1.0, speaker_policy_loss]    
                    
                    # Speaker Entropy loss:
                    speaker_entropy_loss = -torch.cat(self.exp_buffer.speaker_sentences_entrs[:exp_size], dim=-1).permute(0,2,1)
                    # (batch_size, nbr_communication_round, max_sentence_length)
                    speaker_sentences_token_bool = torch.cat(self.exp_buffer.speaker_sentences_token_bool[:exp_size], dim=-1).permute(0,2,1)
                    # (batch_size, nbr_communication_round, max_sentence_length)
                    # Sum on the entropy at each token that are not padding: and average over communication round...
                    speaker_entropy_loss = (speaker_entropy_loss*speaker_sentences_token_bool).sum(-1).mean(-1)
                    # (batch_size, )
                    
                    speaker_entropy_loss_coeff = 0.0
                    if 'max_entr' in config['graphtype']:
                        speaker_entropy_loss_coeff = 1e3
                    losses_dict['speaker_entropy_loss'] = [speaker_entropy_loss_coeff, speaker_entropy_loss]    
                    
                    if exp_size > 1:
                        #TODO: propagate from speaker to listener!!!

                        # Align and reinforce on the listener sentences output:
                        # Each listener sentences contribute to the reward of the next round.
                        # The last listener sentence does not contribute to anything 
                        # (and should not even be computed, as seen by th guard on multi_round above).
                        listener_sentences_log_probs = torch.cat(self.exp_buffer.listener_sentences_log_probs[:exp_size-1], dim=-1)
                        # (batch_size, max_sentence_length, nbr_communication_round-1)
                        # The log likelihood of each sentences is the sum (in log space) over each next token prediction:
                        listener_sentences_log_probs = listener_sentences_log_probs.sum(1)
                        # (batch_size, nbr_communication_round-1)
                        listener_policy_loss = -(listener_sentences_log_probs * (ls[:,1:].detach()-formatted_baseline[:,1:])).sum(dim=-1)
                        # (batch_size, )
                        losses_dict['listener_policy_loss'] = [1.0, listener_policy_loss]    
                    
                        # Listener Entropy loss:
                        listener_entropy_loss = -torch.cat(self.exp_buffer.listener_sentences_entrs[:exp_size], dim=-1).permute(0,2,1)
                        # (batch_size, nbr_communication_round, max_sentence_length)
                        listener_sentences_token_bool = torch.cat(self.exp_buffer.listener_sentences_token_bool[:exp_size], dim=-1).permute(0,2,1)
                        # (batch_size, nbr_communication_round, max_sentence_length)
                        # Sum on the entropy at each token that are not padding: and average over communication round...
                        listener_entropy_loss = (lsitener_entropy_loss*listener_sentences_token_bool).sum(-1).mean(-1)
                        # (batch_size, )
                        
                        listener_entropy_loss_coeff = 0.0
                        if 'max_entr' in config['graphtype']:
                            listener_entropy_loss_coeff = -1e0
                        losses_dict['listener_entropy_loss'] = [listener_entropy_loss_coeff, listener_entropy_loss]    
                    

                    self.exp_buffer.reset()
            else:
                raise NotImplementedError

            
            outputs_dict['decision_probs'] = decision_probs
            
            if 'iterated_learning_scheme' in config \
                and config['iterated_learning_scheme']\
                and 'iterated_learning_rehearse_MDL' in config \
                and config['iterated_learning_rehearse_MDL']:
                # Rehearsing:
                listener_speaking_outputs = self(experiences=sample['speaker_experiences'], 
                                                 sentences=None, 
                                                 multi_round=input_streams_dict['multi_round'],
                                                 graphtype=input_streams_dict['graphtype'],
                                                 tau0=input_streams_dict['tau0'])
                # Let us enforce the Minimum Description Length Principle:
                # Listener's speaking entropy:
                listener_sentences_log_probs = [s_logits.reshape(-1,self.vocab_size).log_softmax(dim=-1) for s_logits in listener_speaking_outputs['sentences_logits']]
                listener_sentences_log_probs = torch.cat(
                    [s_log_probs.gather(dim=-1,index=s_widx[:s_log_probs.shape[0]].long()).sum().unsqueeze(0) 
                    for s_log_probs, s_widx in zip(listener_sentences_log_probs, listener_speaking_outputs['sentences_widx'])], 
                    dim=0)
                listener_entropies_per_sentence = -(listener_sentences_log_probs.exp() * listener_sentences_log_probs)
                # (batch_size, )
                # Maximization:
                losses_dict['ilm_MDL_loss'] = [-config['iterated_learning_rehearse_MDL_factor'], listener_entropies_per_sentence]

                '''
                listener_speaking_entropies = [torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy().mean().view(-1) for w_logits in s_logits], dim=0) for s_logits in listener_speaking_outputs['sentences_logits']]
                # List of size batch_size of Tensor of shape (sentence_length,)
                per_sentence_max_entropies = torch.stack([ lss.max(dim=0)[0] for lss in listener_speaking_entropies])
                # Tensor of shape (batch_size,1)
                ilm_loss = per_sentence_max_entropies.mean(dim=-1)
                # (batch_size, )
                losses_dict['ilm_MDL_loss'] = [1.0, ilm_loss]
                '''

            if 'with_listener_entropy_regularization' in config and config['with_listener_entropy_regularization']:
                entropies = torch.cat([ torch.distributions.categorical.Categorical(logits=d_logits).entropy().view(1) for d_logits in final_decision_logits], dim=-1)
                losses_dict['listener_entropy_loss'] = [config['entropy_regularization_factor'], entropies_per_decision.squeeze()]
                # (batch_size, )
            if config['with_weight_maxl1_loss']:
                losses_dict['listener_maxl1_weight_loss'] = [1.0, weight_maxl1_loss]
        
        # Logging:        
        for logname, value in self.log_dict.items():
            self.logger.add_scalar('{}/{}/{}'.format(mode, self.role, logname), value.item(), it)    
        self.log_dict = {}

        self._tidyup()
        
        outputs_dict['losses'] = losses_dict

        return outputs_dict    