import copy

import ReferentialGym as rg

import torch
import torch.nn as nn 

rg_config = {
    "observability":            "partial",
    "max_sentence_length":      10,
    "nbr_communication_round":  2,
    "nbr_distrators":           3,
    "distractor_sampling":      "uniform",
    "descriptive":              False,
    "object_centric":           False,
    "dynamic_stimulus":         False,

    "algorithm":                "reinforce",# =='categorical' / 'backpropagation' =='gumbel_softmax' or 'straight_through_gumbel_softmax' 
    "vocab_size":               100

    "cultural_pressure_period": None,
}

dataset_args = {
    "dataset_class":            "StimulusDataset",
    "dataset_root_folder":      './test_dataset/',
    "preprocess_function":      'ResizeCNNPreprocessFunction',
}

rgame = rg.make(config=rg_config, dataset=rg_dataset)


# Create agents:
agents = [None] 

config = dict()
config['use_cuda'] = True
config['nbr_stimuli'] = 1 if rg_config['observability'] == "partial" else rg_config['nbr_distrators']+1
config['nbr_frame_stacking'] = 4

# Assuming CNN task:
config['observation_resize_dim'] = 64
# Assuming FC task:
#config['preprocess_function'] = 'PreprocessFunction'

# Model Training Algorithm hyperparameters:
config['model_training_algorithm'] = 'PPO'
# PPO hyperparameters:
config['discount'] = 0.99
config['use_gae'] = True
config['gae_tau'] = 0.95
config['entropy_weight'] = 0.01
config['gradient_clip'] = 5
config['optimization_epochs'] = 10
config['mini_batch_size'] = 128
config['ppo_ratio_clip'] = 0.2
config['learning_rate'] = 3.0e-4
config['adam_eps'] = 1.0e-5

# Recurrent Convolutional Architecture:
config['cnn_encoder_channels'] = [32, 32, 64]
config['cnn_encoder_kernels'] = [6, 4, 3]
config['cnn_encoder_strides'] = [6, 2, 1]
config['cnn_encoder_paddings'] = [0, 1, 1]
config['cnn_encoder_feature_dim'] = 512
config['cnn_encoder_mini_batch_size'] = 128
config['temporal_encoder_nbr_hidden_units'] = 512
config['temporal_encoder_nbr_rnn_layers'] = 1
config['symbol_processing_nbr_hidden_units'] = 512
config['symbol_processing_nbr_rnn_layers'] = 2

listener_config = copy.deepcopy(config)
listener_config['nbr_stimuli'] = rg_config['nbr_stimuli']+1


class BasicCNNGRUSpeaker(rg.Speaker):
    def __init__(self,kwargs, obs_shape, feature_dim=512, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_stimuli, sequence_length, *stimulus_shape)`
                          where, by default, `nbr_stimuli=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param feature_dim: int defining the flatten number of dimension of the features for each stimulus. 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(BasicCNNGRUSpeaker, self).__init__(obs_shape,feature_dim,vocab_size,max_sentence_length)
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        # add the target-boolean-channel:
        cnn_input_shape[0] += 1
        self.cnn_encoder = choose_architecture(architecture='CNN',
                                              input_shape=cnn_input_shape,
                                              hidden_units_list=None,
                                              feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                              nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                              kernels=self.kwargs['cnn_encoder_kernels'],
                                              strides=self.kwargs['cnn_encoder_strides'],
                                              paddings=self.kwargs['cnn_encoder_paddings'])
        
        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        self.temporal_encoder = nn.LSTM(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=0.0,
                                          bidirectional=False)

        symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']+self.kwargs['nbr_stimuli']*self.kwargs['temporal_encoder_nbr_hidden_units']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=0.0,
                                      bidirectional=False)

        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'])
        
    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `stimuli[:, 0]` is assumed as the target stimulus, while the others are distractors, if any. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        batch_size = stimuli.size(0)
        stimuli = stimuli.view(-1, *(stimuli.size()[3:]))
        features = []
        total_size = stimuli.size(0)
        mini_batch_size = self.kwargs['cnn_encoder_mini_batch_size']
        indices = range(0,total_size,mini_batch_size)
        for bidx, eidx in zip(indices,indices[1:]):
            features.append( self.cnn_encoder(stimuli[bidx:eidx]))
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, *self.obs_shape)
        # (batch_size, nbr_stimuli, sequence_length, feature_dim)
        return features 

    def _utter(self, features, sentences=None):
        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_stimuli, sequence_length, feature_dim)
        # Forward pass:
        outputs, next_rnn_states = self.temporal_feature_encoder(features)
        # (batch_size, nbr_stimuli, sequence_length, kwargs['temporal_encoder_nbr_hidden_units'])
        
        embedding_tf_final_outputs = outputs[:,:,-1,:].view(batch_size,1, -1)
        # (batch_size, 1, kwargs['nbr_stimuli'] * kwargs['temporal_encoder_nbr_hidden_units'])
        
        
        if sentences is not None:
            # Consume the sentences:
        
            sentences = sentences.view((-1, self.vocab_size))
            encoded_sentences = self.symbol_encoder(sentences).view((batch_size, self.max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])) 

            states = self.rnn_states
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            inputs = torch.cat( [embedding_tf_final_outputs.repeat(1,self.max_sentence_length,1), encoded_sentences], dim=-1)
            # (batch_size, max_sentence_length, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
            _, (hn,cn) = self.symbol_processing(inputs, states)          
            # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
            
            # Retrieve the current utter_rnn_states as the last state of the symbol processing:
            self.rnn_states = (hn[:,-1,...], cn[:,-1,...])

        # Utter the next sentences:
        next_sentences_one_hots = []
        next_sentences_logits = []
        states = self.rnn_states
        # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
        inputs = torch.zeros((batch_size,1,self.kwargs['symbol_processing_nbr_hidden_units']))
        inputs = torch.cat( [embedding_tf_final_outputs, inputs], dim=-1)
        # (batch_size, 1, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
        # Utter the next sentences:
        for i in range(self.max_sentence_length):
            hiddens, states = self.symbol_processing(inputs, states)          
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            inputs = torch.cat([embedding_tf_final_outputs,hiddens], dim=-1)
            # (batch_size, 1, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']=kwargs['symbol_processing_nbr_hidden_units'])
        
            outputs = self.symbol_decoder(hiddens.squeeze(1))            
            # (batch_size, vocab_size)
            _, prediction = outputs.max(1)                        
            # (batch_size)
            next_sentences_logits.append(outputs.unsqueeze(1))
            next_sentences_one_hot = nn.functional.one_hot(prediction, num_classes=self.vocab_size).unsqueeze(1)
            # (batch_size, 1, vocab_size)
            next_sentences_one_hots.append(next_sentences_one_hot)
        
        self.rnn_states = states 

        next_sentences_one_hots = torch.cat(next_sentences_one_hots, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        next_sentences_logits = torch.cat(symbol_logits, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        return next_sentences_logits, next_sentences_one_hots          
        


class BasicCNNGRUListener(rg.Listener):
    def __init__(self,kwargs, obs_shape, feature_dim=512, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_stimuli, sequence_length, *stimulus_shape)`
                          where, by default, `nbr_stimuli=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param feature_dim: int defining the flatten number of dimension of the features for each stimulus. 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(BasicCNNGRUListener, self).__init__(obs_shape,feature_dim,vocab_size,max_sentence_length)
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        # add the target-boolean-channel:
        cnn_input_shape[0] += 1
        self.cnn_encoder = choose_architecture(architecture='CNN',
                                              input_shape=cnn_input_shape,
                                              hidden_units_list=None,
                                              feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                              nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                              kernels=self.kwargs['cnn_encoder_kernels'],
                                              strides=self.kwargs['cnn_encoder_strides'],
                                              paddings=self.kwargs['cnn_encoder_paddings'])
        
        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        self.temporal_encoder = nn.LSTM(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=0.0,
                                          bidirectional=False)

        symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']+self.kwargs['nbr_stimuli']*self.kwargs['temporal_encoder_nbr_hidden_units']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=0.0,
                                      bidirectional=False)

        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'])

        # Decision making: which input stimuli is the target? 
        self.decision_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.kwargs['nbr_stimuli'])
    
    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        batch_size = stimuli.size(0)
        stimuli = stimuli.view(-1, *(stimuli.size()[3:]))
        features = []
        total_size = stimuli.size(0)
        mini_batch_size = self.kwargs['cnn_encoder_mini_batch_size']
        indices = range(0,total_size,mini_batch_size)
        for bidx, eidx in zip(indices,indices[1:]):
            features.append( self.cnn_encoder(stimuli[bidx:eidx]))
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, *self.obs_shape)
        # (batch_size, nbr_stimuli, sequence_length, feature_dim)
        return features 

    def _reason(self, sentences, features):
        '''
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: None (default) or Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_stimuli, sequence_length, feature_dim)
        # Forward pass:
        outputs, next_rnn_states = self.temporal_feature_encoder(features)
        # (batch_size, nbr_stimuli, sequence_length, kwargs['temporal_encoder_nbr_hidden_units'])
        
        embedding_tf_final_outputs = self.temporalfeatures2states_embedding(outputs[:,:,-1,:].view(batch_size, -1))
        # (batch_size, nbr_stimuli * kwargs['temporal_encoder_nbr_hidden_units']) --> (batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        # Consume the sentences:
        sentences = sentences.view((-1, self.vocab_size))
        encoded_sentences = self.symbol_encoder(sentences).view((batch_size, self.max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])) 

        states = self.rnn_states
        # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
        inputs = torch.cat( [embedding_tf_final_outputs.repeat(1,self.max_sentence_length,1), encoded_sentences], dim=-1)
        # (batch_size, max_sentence_length, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
    
        _, (hn,cn) = self.symbol_processing(inputs, states)          
        # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
        
        # Retrieve the current rnn_states as the last state of the symbol processing:
        self.rnn_states = (hn[:,-1,...], cn[:,-1,...])

        # Compute the decision:
        decision_logits = F.softmax( self.decision_decoder(hn[:,-1,...]))

        return decision_logits


    def _utter(self, features, sentences):
        '''
        Reasons about the features and the listened sentences to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''

        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_stimuli, sequence_length, feature_dim)
        # Forward pass:
        outputs, next_rnn_states = self.temporal_feature_encoder(features)
        # (batch_size, nbr_stimuli, sequence_length, kwargs['temporal_encoder_nbr_hidden_units'])
        
        embedding_tf_final_outputs = outputs[:,:,-1,:].view(batch_size,1, -1)
        # (batch_size, 1, kwargs['nbr_stimuli'] * kwargs['temporal_encoder_nbr_hidden_units'])
        
        
        # No need to consume the sentences:
        # it has been consumed already in the _reason function.
        
        # Utter the next sentences:
        next_sentences_one_hots = []
        next_sentences_logits = []
        states = self.rnn_states
        # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
        inputs = torch.zeros((batch_size,1,self.kwargs['symbol_processing_nbr_hidden_units']))
        inputs = torch.cat( [embedding_tf_final_outputs, inputs], dim=-1)
        # (batch_size, 1, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
        # Utter the next sentences:
        for i in range(self.max_sentence_length):
            hiddens, states = self.symbol_processing(inputs, states)          
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            inputs = torch.cat([embedding_tf_final_outputs,hiddens], dim=-1)
            # (batch_size, 1, kwargs['nbr_stimuli']*kwargs['temporal_encoder_nbr_hidden_units']=kwargs['symbol_processing_nbr_hidden_units'])
        
            outputs = self.symbol_decoder(hiddens.squeeze(1))            
            # (batch_size, vocab_size)
            _, prediction = outputs.max(1)                        
            # (batch_size)
            next_sentences_logits.append(outputs.unsqueeze(1))
            next_sentences_one_hot = nn.functional.one_hot(prediction, num_classes=self.vocab_size).unsqueeze(1)
            # (batch_size, 1, vocab_size)
            next_sentences_one_hots.append(next_sentences_one_hot)
        
        self.rnn_states = states 

        next_sentences_one_hots = torch.cat(next_sentences_one_hots, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        next_sentences_logits = torch.cat(symbol_logits, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        return next_sentences_logits, next_sentences_one_hots



# Train agents:
nbr_epochs = 100
rgame.train(agents=agents, nbr_epochs=nbr_epochs)