from . import datasets
from . import agents
from . import networks

def make(config, dataset_args):
    dataset_args['config'] = config
    
    Dataset = getattr(datasets, dataset_args.pop('dataset_class'))
    
    rg_dataset = Dataset(kwargs=dataset_args)

