from . import modules
from . import datasets
from . import agents
from . import networks

from .referential_game import ReferentialGame

import copy 

def make(config, 
         dataset_args,
         load_path=None,
         save_path=None):
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
    using_v2 = False
    mode2dataset = dataset_args.pop('modes')
    if isinstance(mode2dataset, list):
        using_v2 = True
    
    if using_v2:
        train_dataset = dataset_args["train"]["modes"]["train"]
        need_dict_wrapping = dataset_args["train"]['need_dict_wrapping']
        if "train" in need_dict_wrapping:
            train_dataset = datasets.DictDatasetWrapper(train_dataset)
    else:
        need_dict_wrapping = dataset_args.pop('need_dict_wrapping')
        for key in need_dict_wrapping:
            mode2dataset[key] = datasets.DictDatasetWrapper(mode2dataset[key])
        
        dataset_class = dataset_args.pop('dataset_class', None)
    
        if dataset_class is not None:
            Dataset = getattr(datasets, dataset_class)
        
    rg_datasets = {}
    for mode in mode2dataset:
        if using_v2:
            dataset = dataset_args[mode].pop("modes")[mode]
            need_dict_wrapping = dataset_args[mode].pop('need_dict_wrapping')
            if mode in need_dict_wrapping:
                dataset = datasets.DictDatasetWrapper(dataset)
            
            dataset_class = dataset_args[mode].pop('dataset_class', None)
            if dataset_class is not None:
                Dataset = getattr(datasets, dataset_class)    
        else:
            dataset = mode2dataset[mode]

        ###

        if Dataset is None:
            rg_datasets[mode] = dataset
        else:
            if using_v2:
                inner_dataset_args = copy.deepcopy(dataset_args[mode])
            else:
                inner_dataset_args = copy.deepcopy(dataset_args)
            
            if dataset_class == 'LabeledDataset': 
                inner_dataset_args['dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)
            elif dataset_class == 'DualLabeledDataset':
                if using_v2:
                    inner_dataset_args['train_dataset'] = train_dataset
                else:
                    inner_dataset_args['train_dataset'] = mode2dataset["train"]
                inner_dataset_args['test_dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)

    modules = config.pop("modules")
    pipelines = config.pop("pipelines")
    rg_instance = ReferentialGame(
        datasets=rg_datasets, 
        config=config,
        modules=modules,
        pipelines=pipelines,
        load_path=load_path,
        save_path=save_path,
        use_priority=config['use_priority'] if 'use_priority' in config else False,
    )
    
    return rg_instance
