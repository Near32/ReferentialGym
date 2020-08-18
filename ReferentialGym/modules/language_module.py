from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F 

from .module import Module
from ..networks import layer_init

def build_LanguageModule(id:str,
                       config:Dict[str,object],
                       input_stream_ids:Dict[str,str]) -> Module:
    
    use_sentences_one_hot_vectors = config['use_sentences_one_hot_vectors']
    
    if use_sentences_one_hot_vectors:
        # Assumes vocab_size is given WITH SoS, EoS, and PAD here.
        symbol_embedding = nn.Sequential(
            layer_init(
                nn.Linear(config['vocab_size'], config['symbol_embedding_size'], bias=False)
            ),
            nn.Dropout(p=config['embedding_dropout_prob'])
        )
    else:
        # Assumes vocab_size is given WITHOUT SoS, EoS, and PAD at the end here.
        symbol_embedding = nn.Embedding(config['vocab_size']+3, config['symbol_embedding_size'], padding_idx=config['vocab_size']+3)
        
    
    rnn_type = config['rnn_type']
    if 'lstm' in self.rnn_type:
        symbol_processing = nn.LSTM(
            input_size=config['symbol_embedding_size'],
            hidden_size=config['symbol_processing_nbr_hidden_units'], 
            num_layers=config['symbol_processing_nbr_rnn_layers'],
            batch_first=True,
            dropout=config['processing_dropout_prob'],
            bidirectional=False
        )
    elif 'gru' in self.rnn_type:
        symbol_processing = nn.GRU(
            input_size=config['symbol_embedding_size'],
            hidden_size=config['symbol_processing_nbr_hidden_units'], 
            num_layers=config['symbol_processing_nbr_rnn_layers'],
            batch_first=True,
            dropout=config['processing_dropout_prob'],
            bidirectional=False
        )
    else:
        raise NotImplementedError
    
    module = LanguageModule(
        id=id,
        embedding=symbol_embedding,
        processing=symbol_processing,
        config=config,
        input_stream_ids=input_stream_ids
    )

    print(module)
    
    return module

class LanguageModule(Module):
    def __init__(self, 
                 id, 
                 embedding,
                 processing,
                 config,
                 input_stream_ids):
        
        assert "inputs" in input_stream_ids.keys(),\
               "LanguageModule relies on 'inputs' id to start its pipeline.\n\
                Not found in input_stream_ids."
        assert "losses_dict" in input_stream_ids.keys(),\
               "LanguageModule relies on 'losses_dict' id to record the computated losses.\n\
                Not found in input_stream_ids."
        assert "logs_dict" in input_stream_ids.keys(),\
               "LanguageModule relies on 'logs_dict' id to record the accuracies.\n\
                Not found in input_stream_ids."
        assert "mode" in input_stream_ids.keys(),\
               "LanguageModule relies on 'mode' key to record the computated losses and accuracies.\n\
                Not found in input_stream_ids."

        
        super(LanguageModule, self).__init__(id=id, 
            type="LanguageModule", 
            config=config, 
            input_stream_ids=input_stream_ids)
        
        self.kwargs = config
        
        self.symbol_embedding = embedding 
        self.symbol_processing = processing

        self.vocab_size = config['vocab_size']
        if not config['use_sentences_one_hot_vectors']:
            # Account for SoS, EoS, and PAD:
            self.vocab_size += 3
        
        """
        self.symbol_processing_learnable_initial_state = nn.Parameter(
                torch.zeros(1,1,self.kwargs['symbol_processing_nbr_hidden_units'])
        )
        """
        
        if self.config["use_cuda"]:
            self = self.cuda()
    
    def embed_sentences(self, sentences):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :returns embedded_sentences: Tensor of shape `(batch_size, max_sentence_length, symbol_embedding_size)` containing the padded sequence of embedded symbols.
        """
        batch_size = sentences.shape[0]
        # (batch_size, max_sentence_length, self.vocab_size)
        sentences = sentences.view((-1, self.vocab_size)).float()
        embedded_symbols = self.symbol_embedding(sentences) 
        # (batch_size*max_sentence_length, self.kwargs['symbol_embedding_size'])
        embedded_sentences = embedded_symbols.view((batch_size, -1, self.kwargs['symbol_embedding_size']))
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        return embedded_sentences

    def find_sentence_lengths(self, sentences):
        """
        Adapted from:
        https://github.com/facebookresearch/EGG/blob/2e2d42e73f50af0ce70ab22e1ff77bf3a38ab6ef/egg/core/util.py#L267

        :param sentences:   Tensor of shape `(batch_size, max_sentence_length, vocab_size)` 
                            containing one-hot-encoded symbols.
        :returns: Tensor of shape `(batch_size,)` containing lengths of sentences until the first EoS token, included!
                    NOTE: we include the EoS token to guarantee non-negative sentence lenghts...

        """
        # Assumes EoS idx==0:
        sentences_token_idx = (sentences*torch.arange(self.vocab_size).to(sentences.device)).sum(-1)
        #(batch_size, max_sentence_length)
        eos_mask = (sentences_token_idx==0)
        lengths = ((eos_mask.cumsum(-1)>0)<=0).sum(-1)
        #(batch_size, )
        return lengths.add_(1).clamp_(max=self.max_sentence_length)

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        mode = input_streams_dict["mode"]
        sentences = input_streams_dict["inputs"]
        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]

        batch_size = sentences.size(0)
        
        # Consume the sentences:
        # (batch_size, max_sentence_length, self.vocab_size)
        embedded_sentences = self.embed_sentences(sentences)
        # (batch_size, max_sentence_length, self.kwargs['symbol_embedding_size'])
        
        
        if self.kwargs['use_pack_padding']:
            sentence_lengths = self.find_sentence_lengths(sentences)
            #(batch_size, )
        
            symbol_processing_input = nn.utils.rnn.pack_padded_sequence(
                embedded_sentences, 
                sentence_lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
        else:
            symbol_processing_input = embedded_sentences
        
        states = None
        rnn_outputs, self.rnn_states = self.symbol_processing(symbol_processing_input, states)    
        
        if self.kwargs['use_pack_padding']:
            rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs,batch_first=True)
        # (batch_size, ?max_sentence_length?, kwargs['symbol_processing_nbr_hidden_units'])
        # Watchout for max_sentence_length when using pack_padding, it may be lower than expected value.
        # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
        outputs_stream_dict["rnn_states"] = self.rnn_states
        outputs_stream_dict["rnn_outputs"] = rnn_outputs
        
        outputs_stream_dict["final_rnn_outputs"] = rnn_outputs[:,-1,...]
        # (batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        
    
        return outputs_stream_dict 