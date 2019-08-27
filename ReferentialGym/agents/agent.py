import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class Agent(nn.Module):
    def __init__(self, agent_id='l0', logger=None):
        """
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        """
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.logger = logger 
        self.log_idx = 0

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

    def _log(self, log_dict):
        if self.logger is None: 
            return 

        agent_log_dict = {f"{self.agent_id}": dict()}
        for key, data in log_dict.items():
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

    def train(self, inputs_dict, config, role='speaker'):
        """
        Compute the losses and return them along with the produced outputs.

        :param inputs_dict: Dict that should contain, at least, the following keys and values:
            - `'sentences'`: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
            - `'experiences'`: Tensor of shape `(batch_size, *self.obs_shape)`. 
                            Make sure to shuffle the experiences so that the order does not give away the target. 
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

        outputs_dict = self(sentences=inputs_dict['sentences'],
                           experiences=inputs_dict['experiences'],
                           multi_round=inputs_dict['multi_round'],
                           graphtype=inputs_dict['graphtype'],
                           tau0=inputs_dict['tau0'])


        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        # //------------------------------------------------------------//
        

        weight_maxl1_loss = 0.0
        for p in self.parameters() :
            weight_maxl1_loss += torch.max( torch.abs(p) )
        outputs_dict['maxl1_loss'] = weight_maxl1_loss        

        losses_dict = dict()
        if 'speaker' in role:
            if config['with_utterance_penalization'] or config['with_utterance_promotion']:
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
                #oov_loss = -(config['utterance_factor']/(batch_size*config['max_sentence_length']))*torch.sum(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
                oov_loss = -(1.0/(batch_size*config['max_sentence_length']))*torch.sum(speaker_utterances_count*torch.log(d_speaker_utterances_probs+1e-10))
                #oov_loss = -(self.config['utterance_penalization_factor']/(batch_size*self.config['max_sentence_length']))*torch.sum(listener_speaking_utterances_count*d_listener_speaking_utterances_probs)
                if config['with_utterance_promotion']:
                    oov_loss *= -1 
                losses_dict['oov_loss'] = [config['utterance_factor'], oov_loss]

            if config['with_mdl_principle']:
                speaker_utterances_max_logit = torch.cat([torch.max(s, dim=-1)[0].view((-1)) for s in outputs_dict['sentences_logits']], dim=0)
                #speaker_utterances_max_logit = torch.cat([torch.max(s[0], dim=-1)[0].view((-1)) for s in outputs_dict['sentences_logits']], dim=0)
                mask = (speaker_utterances_max_logit < config['obverter_stop_threshold']).float()
                if speaker_utterances_max_logit.is_cuda: mask = mask.cuda()
                # (batch_size*sentence_length,)
                #mdl_loss = -(mask*speaker_utterances_max_logit).sum()/(batch_size*self.config['max_sentence_length'])
                mdl_loss = (mask*(1-speaker_utterances_max_logit)).sum()/(batch_size)
                losses['mdl_loss'] = (config['mdl_principle_factor'], mdl_loss)
            
            if config['with_speaker_entropy_regularization']:
                entropies = torch.cat([torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy() for w_logits in s_logits], dim=0) for s_logits in outputs_dict['sentences_logits']], dim=0)
                losses_dict['speaker_entropy_regularization_loss'] = [config['entropy_regularization_factor'], entropies.mean()]

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
            if config['descriptive']:  
                decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                criterion = nn.NLLLoss(reduction='mean')
                
                if config['obverter_least_effort_loss']:
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
            else:   
                final_decision_logits = final_decision_logits[:,-1,...]
                # (batch_size, (nbr_distractors+1) / ? (descriptive mode depends on the role of the agent) )
                decision_probs = F.log_softmax( final_decision_logits, dim=-1)
                criterion = nn.NLLLoss(reduction='mean')
                loss = criterion( final_decision_logits, sample['target_decision_idx'])
            losses_dict['referential_game_loss'] = [1.0, loss] 

            #Havrylov's Hinge loss:
            '''
            losses = []
            for b in range(final_decision_logits.size(0)):
                fd_target_l = final_decision_logits[b,sample['target_decision_idx'][b]]
                bl = 0.0
                for d in range(final_decision_logits.size(1)):
                    if d == sample['target_decision_idx'][b]: continue
                    el = 1-fd_target_l+final_decision_logits[b][d]
                    el = torch.max(torch.zeros_like(el),el)
                    bl = bl + el 
                losses.append( bl.unsqueeze(0))
            losses = torch.cat(losses,dim=0)
            loss = losses.mean()
            '''

            outputs_dict['decision_probs'] = decision_probs
            
            if config['iterated_learning_scheme']:
                # Let us enforce the Minimum Description Length Principle:
                # Listener's speaking entropy:
                listener_speaking_entropies = [torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=w_logits).entropy().mean().view(-1) for w_logits in s_logits], dim=0) for s_logits in listener_speaking_outputs['sentences_logits']]
                # List of size batch_size of Tensor of shape (sentence_length,)
                per_sentence_max_entropies = torch.stack([ lss.max(dim=0)[0] for lss in listener_speaking_entropies])
                # Tensor of shape (batch_size,1)
                ilm_loss = per_sentence_max_entropies.mean()
                losses_dict['ilm_loss'] = [1.0, ilm_loss]            

            if config['with_listener_entropy_regularization']:
                entropies = torch.cat([ torch.distributions.bernoulli.Bernoulli(logits=d_logits).entropy() for d_logits in final_decision_logits], dim=0)
                losses_dict['listener_entropy_loss'] = [config['entropy_regularization_factor'], entropies.mean()]

            if config['with_weight_maxl1_loss']:
                losses_dict['listener_maxl1_weight_loss'] = [1.0, weight_maxl1_loss]
        
        return outputs_dict, losses_dict    