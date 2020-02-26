from . import datasets
from . import agents
from . import networks
from .referential_game import ReferentialGame

def make(config, dataset_args):
    """
    TODO
    """

    train_dataset = dataset_args.pop('train_dataset')
    test_dataset = dataset_args.pop('test_dataset')

    if dataset_args['dataset_class'] is not None:
        dataset_class = dataset_args.pop('dataset_class')
        Dataset = getattr(datasets, dataset_class)
        
        if dataset_class == 'LabeledDataset': 
            dataset_args['dataset'] = train_dataset
            rg_train_dataset = Dataset(kwargs=dataset_args)

            dataset_args['dataset'] = test_dataset
            rg_test_dataset = Dataset(kwargs=dataset_args)
        elif dataset_class == 'DualLabeledDataset':
            dataset_args['train_dataset'] = train_dataset
            dataset_args['test_dataset'] = test_dataset
            dataset_args['mode'] = 'train'
            rg_train_dataset = Dataset(kwargs=dataset_args)

            dataset_args['mode'] = 'test'
            rg_test_dataset = Dataset(kwargs=dataset_args)
    else:
        rg_train_dataset = train_dataset
        rg_test_dataset = test_dataset
        
    rg_datasets = {'train':rg_train_dataset, 'test':rg_test_dataset}
    rg_instance = ReferentialGame(datasets=rg_datasets, config=config)

    return rg_instance

