import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import HomoscedasticMultiTasksLoss
import copy


class Agent(nn.Module):
    def __init__(self, agent_id='l0', logger=None, kwargs=None):
        """
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        """
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.logger = logger 
        self.kwargs = kwargs
        self.log_idx = 0
        self.log_dict = dict()

        self.use_sentences_one_hot_vectors = False

        self.homoscedastic = self.kwargs['homoscedastic_multitasks_loss']
        if self.homoscedastic:
            self.homoscedastic_speaker_loss = HomoscedasticMultiTasksLoss()
            self.homoscedastic_listener_loss = HomoscedasticMultiTasksLoss()

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

        agent_log_dict = {f"{self.agent_id}": dict()}
        for key, data in log_dict.items():
            if data is None:
                data = [None]*batch_size
            agent_log_dict[f"{self.agent_id}"].update({f"{key}":data})
        
        self.logger.add_dict(agent_log_dict, batch=True, idx=self.log_idx) 
        
        self.log_idx += 1

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

    def compute(self, inputs_dict, config, role='speaker', mode='train', it=0):
        """
        Compute the losses and return them along with the produced outputs.

        :param inputs_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences_widx'`: Tensor of shape `(batch_size, max_sentence_length, 1)` containing the padded sequence of symbols' indices.
            - `'sentences_one_hot'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
            - `'latent_experiences'`: Tensor of shape `(batch_size, nbr_latent_dimensions)`.
            - `'multi_round'`: Boolean defining whether to utter a sentence back or not.
            - `'graphtype'`: String defining the type of symbols used in the output sentence:
                        - `'categorical'`: one-hot-encoded symbols.
                        - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                        - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                        - `'obverter'`: obverter training scheme...
            - `'tau0'`: Float, temperature with which to apply gumbel-softmax estimator. 
            - `'sample'`: Dict that contains the speaker and listener experiences as well as the target index.
        """
        
        batch_size = len(inputs_dict['experiences'])

        input_sentence = inputs_dict['sentences_widx']
        if self.use_sentences_one_hot_vectors:
            input_sentence = inputs_dict['sentences_one_hot']

        outputs_dict = self(sentences=input_sentence,
                           experiences=inputs_dict['experiences'],
                           multi_round=inputs_dict['multi_round'],
                           graphtype=inputs_dict['graphtype'],
                           tau0=inputs_dict['tau0'])

        outputs_dict['latent_experiences'] = inputs_dict['latent_experiences']
        self._log(outputs_dict, batch_size=batch_size)

        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        

        weight_maxl1_loss = 0.0
        for p in self.parameters() :
            weight_maxl1_loss += torch.max( torch.abs(p) )
        outputs_dict['maxl1_loss'] = weight_maxl1_loss        


        for logname, value in self.log_dict.items():
            self.logger.add_scalar('{}/{}/{}'.format(mode,role, logname), value.item(), it)    
        self.log_dict = {}


        losses_dict = dict()

        '''
        if hasattr(self, 'TC_losses'):
            losses_dict[f'{role}/TC_loss'] = [1.0, self.TC_losses]
        '''
        if hasattr(self, 'VAE_losses'):# and ('listener' in role or not('obverter' in inputs_dict['graphtype'])):
            losses_dict[f'{role}/VAE_loss'] = [1.0, self.VAE_losses]

        if 'speaker' in role:
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
        

        if 'listener' in role:
            sample = inputs_dict['sample']
            if config['iterated_learning_scheme']:
                listener_speaking_outputs = self(experiences=sample['speaker_experiences'], 
                                                 sentences=None, 
                                                 multi_round=inputs_dict['multi_round'],
                                                 graphtype=inputs_dict['graphtype'],
                                                 tau0=inputs_dict['tau0'])
            
            
            decision_logits = outputs_dict['decision']
            final_decision_logits = decision_logits
            # (batch_size, max_sentence_length / squeezed if not using obverter agent, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
            if config['agent_loss_type'] == 'NLL':
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
            elif config['agent_loss_type'] == 'Hinge':
                #Havrylov's Hinge loss:
                final_decision_logits = final_decision_logits[:,-1,...]
                # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
                decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                
                target_decision_idx = sample['target_decision_idx'].unsqueeze(1)
                target_decision_logits = final_decision_logits.gather(dim=1, index=target_decision_idx)
                # (batch_size, 1)

                distractors_logits_list = [torch.cat([pb_dl[:tidx.item()], pb_dl[tidx.item()+1:]], dim=0).unsqueeze(0) 
                    for pb_dl, tidx in zip(final_decision_logits, target_decision_idx)]
                distractors_decision_logits = torch.cat(
                    distractors_logits_list, 
                    dim=0)
                # (batch_size, nbr_distractors)
                
                loss_element = 1-target_decision_logits+distractors_decision_logits
                # (batch_size, nbr_distractors)
                maxloss_element = torch.max(torch.zeros_like(loss_element), loss_element)
                loss = maxloss_element.sum(dim=1)
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

            
            outputs_dict['decision_probs'] = decision_probs
            
            if 'iterated_learning_scheme' in config and config['iterated_learning_scheme']:
                # Let us enforce the Minimum Description Length Principle:
                # Listener's speaking entropy:
                listener_speaking_entropies = [torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy().mean().view(-1) for w_logits in s_logits], dim=0) for s_logits in listener_speaking_outputs['sentences_logits']]
                # List of size batch_size of Tensor of shape (sentence_length,)
                per_sentence_max_entropies = torch.stack([ lss.max(dim=0)[0] for lss in listener_speaking_entropies])
                # Tensor of shape (batch_size,1)
                ilm_loss = per_sentence_max_entropies.mean(dim=-1)
                # (batch_size, )
                losses_dict['ilm_loss'] = [1.0, ilm_loss]            

            if 'with_listener_entropy_regularization' in config and config['with_listener_entropy_regularization']:
                entropies = torch.cat([ torch.distributions.categorical.Categorical(logits=d_logits).entropy().view(1) for d_logits in final_decision_logits], dim=-1)
                losses_dict['listener_entropy_loss'] = [config['entropy_regularization_factor'], entropies_per_decision.squeeze()]
                # (batch_size, )
            if config['with_weight_maxl1_loss']:
                losses_dict['listener_maxl1_weight_loss'] = [1.0, weight_maxl1_loss]
        
        if len(losses_dict) and self.homoscedastic:
            if 'speaker' in role:   losses_dict = self.homoscedastic_speaker_loss(losses_dict)
            if 'listener' in role:   losses_dict = self.homoscedastic_listener_loss(losses_dict)

        self._tidyup()
        
        return outputs_dict, losses_dict    