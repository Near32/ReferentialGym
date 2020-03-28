import os
import pickle 
import torch
import numpy as np 
from tqdm import tqdm 

from tensorboardX import SummaryWriter
from ..utils import compute_topographic_similarity_parallel

class statsLogger(SummaryWriter):
    def __init__(self,path='./',filename='logs.dict',dumpPeriod=1):
        """
        
        :param path: `str` defining the folder path where to dump the data and save the tensorboard summaries.
        :param filename: `str` defining the name of the file where the data are dumped.
        :param dumpPeriod: `in` defining the period with which we dump the data.
        """
        super(statsLogger, self).__init__(path)

        self.path = path
        self.filename = filename
        self.dumpIdx = 0
        self.dumpPeriod = dumpPeriod
        self.counterDumpPeriod = 0

        if not os.path.exists(self.path) :
            os.mkdir(self.path)

        self.data = [dict()]
            
    def dump(self) :
        """
        Dumps the data that have been recorder so far with the `add_dict` method.
        """
        '''
        file = open(os.path.join(self.path, self.filename+f".{self.dumpIdx}"), "wb")
        pickle.dump(self.data, file)
        file.close()
        '''

        self.dumpIdx += 1
        self.data = [dict() if idx!=(len(self.data)-1) else d for idx, d in enumerate(self.data)]

    def switch_epoch(self,):
        """
        Separate the data recording in epoch-based entries.
        """
        self.dump()
        self.data.append(dict())

    def measure_topographic_similarity(self, 
                                       sentences_key='sentences_widx',
                                       features_key='temporal_features',
                                       max_nbr_samples=None,
                                       comprange=None,
                                       verbose=True,
                                       max_workers=16):
        '''
        Accounts for a measure of the compositionality of the current epoch-like set of data,
        following the computation of topographic similarity.
        '''
        data = self.data[-1]
        cleaned_data = dict()

        # Remove None items and set up the data in a list:
        for agent_id in data:
            if not(sentences_key in data[agent_id]): continue
            if not(features_key in data[agent_id]): continue
            
            for idx, sd in enumerate(data[agent_id][sentences_key]):
                if sd[1] is None: continue
                if not(agent_id in cleaned_data): cleaned_data[agent_id] = dict()

                if not(sentences_key in cleaned_data[agent_id]):  cleaned_data[agent_id][sentences_key] = list()
                cleaned_data[agent_id][sentences_key].append( {'id':agent_id, 'it':sd[0], 'raw':sd[1]})

                fd = data[agent_id][features_key][idx]
                assert(fd[0] == sd[0])
                
                if not(features_key in cleaned_data[agent_id]):  cleaned_data[agent_id][features_key] = list()
                cleaned_data[agent_id][features_key].append( {'id':agent_id, 'it':fd[0], 'raw':fd[1]})

        rhos = dict()
        ps = dict()        
        unique_prod_ratios = dict()
        for agent_id in cleaned_data:    
            if max_nbr_samples is None: max_nbr_samples = len(cleaned_data[agent_id][sentences_key])

            it_sentences = [ (s['it'],s['raw']) for s in cleaned_data[agent_id][sentences_key][:max_nbr_samples]]
            sentences = [s[1] for s in it_sentences]

            features = []
            sentence_idx = 0 
            for idx, fd in enumerate(data[agent_id][features_key]):
                it, tf = fd[0], fd[1]
                if len(features) == len(sentences): 
                        break
                
                sentence_it = it_sentences[sentence_idx][0]
                if sentence_it == it:
                    features.append(tf.reshape(-1))
                    sentence_idx += 1

            np_sentences = np.concatenate( sentences, axis=1).transpose(1,0)

            _, idx_unique_sentences = np.unique(np_sentences, axis=0, return_index=True)
            idx_unique_sentences = sorted(idx_unique_sentences)

            unique_it_sentences = [ it_sentences[idx] for idx in idx_unique_sentences]
            unique_sentences = [s[1] for s in unique_it_sentences]

            unique_sentences_features = [ fd['raw'] for idx, fd in enumerate(cleaned_data[agent_id][features_key][:max_nbr_samples]) if idx in idx_unique_sentences]
            #features = [ tf for idx, (it, tf) in enumerate(data[agent_idx][sentences_key]) if idx in idx_unique_sentences]

            if verbose:
                print("Agent {} :: There are {} unique sentences out of the {} sampled sentences.".format(agent_id, len(unique_sentences), len(it_sentences)))

            if comprange is None: comprange = max_nbr_samples
            
            rho, p, levs, cossims = compute_topographic_similarity_parallel(sentences=unique_sentences, 
                                                                            features=unique_sentences_features, 
                                                                            comprange=comprange,
                                                                            max_workers=max_workers)

            rhos[agent_id] = rho 
            ps[agent_id] = p
            unique_prod_ratios[agent_id] = len(unique_sentences) / len(np_sentences) * 100.0

        return rhos, ps, unique_prod_ratios
        

    def add_dict(self,x, rec=None, batch=False, idx=None) :
        """
        Records the data.

        :param x: `dict` of data to record. If it is a imbrigated dictionnary,
                  then it is recorded with the same hierarchy.
        :param rec: `None` or `dict` of the current level in the recording hierarchy.
        :param batch: boolean that defines whether to the data to be recorded is given as a batch or not.
        """
        # When this function is called from another object,
        # we use the latest period-associated entry in the data recording:
        if rec is None: 
            rec = self.data[-1]
            # Shall we dump the data recorder so far?
            self.counterDumpPeriod += 1
            if self.counterDumpPeriod%self.dumpPeriod == 0 :
                self.dump()
        
        # Otherwise it is called recursively by itself:
        elif not(isinstance(x, dict)):
            # Rec might be a dictionnary, newly created:
            if isinstance(rec, dict): rec = list()
            if x is None: batch = False 
            if batch:
                for bidx in range(len(x)):
                    xin = x[bidx]
                    if isinstance(xin, torch.Tensor):   xin = xin.cpu().detach().numpy()
                    rec.append((idx,xin))
            else:
                if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
                rec.append((idx,x))
            
            return rec

        # If the data to record, i.e. x, is a dictionnary to expand,
        # then we keep on recursively creating new dictionnaries:
        for key in x.keys() :
            if not(key in rec.keys()) :
                rec[key] = dict()
            rec[key] = self.add_dict(x=x[key], rec=rec[key], batch=batch, idx=idx)
        return rec 
