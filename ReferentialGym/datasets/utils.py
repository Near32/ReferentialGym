import torch

class DictBatch:
    def __init__(self, data):
        """
        :param data: list of Dict of Tensors.
        """
        self.keys = list(data[0].keys())
        values = list(zip(*[list(d.values()) for d in data]))

        for idx, key in enumerate(self.keys):
            setattr(self, key, torch.cat(values[idx], dim=0))
        
    def pin_memory(self):
        for key in self.keys:
            attr = getattr(self, key).pin_memory()
            setattr(self, key, attr)
        return self

    def cuda(self):
        for key in self.keys:
            attr = getattr(self, key).cuda()
            setattr(self, key, attr)
        return self

    def __getitem__(self, key):
        """
        :param key: str
        """
        return getattr(self, key, None)

def collate_dict_wrapper(batch):
    return DictBatch(batch)