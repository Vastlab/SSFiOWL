import torch
import random
import numpy as np
from vast.tools import logger as vastlogger
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def random_selector(features, classes_of_interest, no_of_exemplars=None):
    logger = vastlogger.get_logger()
    exemplars_to_return = {}
    for cls_name in classes_of_interest:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        ind_of_interest = torch.randint(features[cls_name].shape[0],
                                        (min(no_of_exemplars,
                                             features[cls_name].shape[0]),
                                         1))
        current_exemplars = features[cls_name].gather(
                                            0, ind_of_interest.expand(-1, features[cls_name].shape[1]))
        exemplars_to_return[cls_name]=current_exemplars
    return exemplars_to_return

def add_all_negatives(features, classes_of_interest,no_of_exemplars=None):
    exemplars_to_return = {}
    for cls in classes_of_interest:
        exemplars_to_return[cls] = features[cls]
    return exemplars_to_return