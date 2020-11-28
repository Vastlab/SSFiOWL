import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def random_selector(features, rolling_models,no_of_exemplars=None):
    print(f"Randomly Selecting extreme vectors and adding them to exemplars")
    current_batch_size = 0
    exemplars_to_return = {}
    no_of_exemplar_batches = 0
    exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = []
    for cls_name in rolling_models:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        ind_of_interest = torch.randint(rolling_models[cls_name]['extreme_vectors'].shape[0],
                                        (min(no_of_exemplars,
                                             rolling_models[cls_name]['extreme_vectors'].shape[0]),
                                         1))
        current_exemplars = features[cls_name]['features'].gather(0, ind_of_interest.expand(
            -1, rolling_models[cls_name]['extreme_vectors'].shape[1]))
        exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'].append(current_exemplars)
        current_batch_size += current_exemplars.shape[0]
        if current_batch_size >= 1000:
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(
                exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'])
            no_of_exemplar_batches += 1
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = []
            current_batch_size = 0
    if type(exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']) == list:
        if len(exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']) == 0:
            del exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']
        else:
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(
                exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'])
            no_of_exemplar_batches += 1
    return exemplars_to_return



def add_all_negatives(features, rolling_models,no_of_exemplars=None):
    no_of_exemplar_batches = 0
    current_batch_size = 0
    exemplars_to_return = {}
    exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = []
    for cls in sorted(set(rolling_models.keys())):
        current_exemplars = features[cls]['features']
        exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'].append(current_exemplars)
        current_batch_size += current_exemplars.shape[0]
        if current_batch_size >= 1000:
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(
                exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'])
            no_of_exemplar_batches += 1
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = []
            current_batch_size = 0
    if type(exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']) == list:
        if len(exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']) == 0:
            del exemplars_to_return[f'exemplars_{no_of_exemplar_batches}']
        else:
            exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(
                exemplars_to_return[f'exemplars_{no_of_exemplar_batches}'])
    return exemplars_to_return