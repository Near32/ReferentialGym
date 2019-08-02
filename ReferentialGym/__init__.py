from . import datasets
from . import agents
from . import networks
from .referential_game import ReferentialGame

def make(config, dataset_args):
    Dataset = getattr(datasets, dataset_args.pop('dataset_class'))
    rg_dataset = Dataset(kwargs=dataset_args)

    rg_instance = ReferentialGame(dataset=rg_dataset, config=config)

    return rg_instance

