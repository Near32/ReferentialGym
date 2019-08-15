import os
import pickle 
import torch

from tensorboardX import SummaryWriter

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
        self.dumpPeriod = dumpPeriod
        self.counterDumpPeriod = 0

        if not os.path.exists(self.path) :
            os.mkdir(self.path)

        self.data = [dict()]
        
        if self.filename in os.listdir(self.path) :
            file = open(os.path.join(self.path, self.filename), "rb")
            self.data = pickle.load(file)
            file.close()
            
    def dump(self) :
        """
        Dumps the data that have been recorder so far with the `add_dict` method.
        """
        file = open(os.path.join(self.path, self.filename+f".{self.counterDumpPeriod//self.dumpPeriod-1}"), "wb")
        pickle.dump(self.data, file)
        file.close()
        
        self.data = [dict() if idx!=(len(self.data)-1) else d for idx, d in enumerate(self.data)]

    def switch_epoch(self,):
        """
        Separate the data recording in epoch-based entries.
        """
        self.data.append(dict())

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