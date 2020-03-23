from . import datasets
from . import agents
from . import networks
from .referential_game import ReferentialGame

import copy 


def make(config, dataset_args):
    """
    Create a ReferentialGame with all the different evalutation modes,
    that are specified by the `dataset_args`'s `mode` entry.
    :param config: Dict that specifies all the important hyperparameters of the game.
    :param dataset_args: Dict with the following expected entries:
        - `dataset_class`: None, `'LabeledDataset'`, or `'DualLabeledDataset' is expected.
                           It specifies the class of dataset decorator to use.
        - `modes`: Dict of training/evaluation mode as keys and corresponding datasets as values.
                   `'test'` and `'train'` are mandatory.
    """
    dataset_class = dataset_args.pop('dataset_class')
    if dataset_class is not None:
        Dataset = getattr(datasets, dataset_class)
    
    modes = dataset_args.pop('modes')
    train_dataset = modes['train']
    test_dataset = modes['test']

    rg_datasets = {}
    for mode, dataset in modes.items():
        if Dataset is None:
            rg_datasets[mode] = dataset
        else:
            inner_dataset_args = copy.deepcopy(dataset_args)
            if dataset_class == 'LabeledDataset': 
                inner_dataset_args['dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)
            elif dataset_class == 'DualLabeledDataset':
                inner_dataset_args['train_dataset'] = train_dataset
                inner_dataset_args['test_dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)

    rg_instance = ReferentialGame(datasets=rg_datasets, config=config)
    return rg_instance

    '''
    train_dataset = dataset_args.pop('train_dataset')
    test_dataset = dataset_args.pop('test_dataset')

    if dataset_args['dataset_class'] is not None:
        dataset_class = dataset_args.pop('dataset_class')
        Dataset = getattr(datasets, dataset_class)
        
        if dataset_class == 'LabeledDataset': 
            dataset_args['dataset'] = train_dataset
            dataset_args['mode'] = 'train'
            rg_train_dataset = Dataset(kwargs=dataset_args)

            dataset_args['dataset'] = test_dataset
            dataset_args['mode'] = 'test'
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
    '''
