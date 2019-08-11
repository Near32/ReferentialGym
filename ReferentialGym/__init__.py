from . import datasets
from . import agents
from . import networks
from .referential_game import ReferentialGame

def make(config, dataset_args):
    """Test Sphinx Documentation.


    """

    Dataset = getattr(datasets, dataset_args.pop('dataset_class'))
    
    train_dataset = dataset_args.pop('train_dataset')
    test_dataset = dataset_args.pop('test_dataset')

    dataset_args['dataset'] = train_dataset
    rg_train_dataset = Dataset(kwargs=dataset_args)

    dataset_args['dataset'] = test_dataset
    rg_test_dataset = Dataset(kwargs=dataset_args)

    rg_datasets = {'train':rg_train_dataset, 'test':rg_test_dataset}
    rg_instance = ReferentialGame(datasets=rg_datasets, config=config)

    return rg_instance

