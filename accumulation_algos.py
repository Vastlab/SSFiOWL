import torch

from vast.tools import logger as vastlogger
logger = vastlogger.get_logger()

def find_unknowness_probabilities(probabilities_for_train_set, unknowness_threshold=None):
    unknowness_scores = {}
    for k in set(probabilities_for_train_set.keys())-set(['classes_order']):
        unknowness_scores[k] = 1-torch.max(probabilities_for_train_set[k],dim=1).values
        if unknowness_threshold is not None:
            unknowness_scores[k] = unknowness_scores[k]>unknowness_threshold
    return unknowness_scores


@vastlogger.time_recorder
def mimic_incremental(args, current_batch,rolling_models, probabilities_for_train_set):
    accumulated_samples = {}
    for c in current_batch:
        if c not in rolling_models:
            accumulated_samples[c] = current_batch[c]
    return accumulated_samples


@vastlogger.time_recorder
def learn_new_unknowns(args, operating_batch, rolling_models, probabilities_for_train_set):
    if len(rolling_models.keys())==0:
        return operating_batch
    unknowness_scores = find_unknowness_probabilities(probabilities_for_train_set,
                                                      unknowness_threshold=args.unknowness_threshold)
    accumulated_samples = {}
    class_names = sorted(list(set(operating_batch.keys())))
    for cls in class_names:
        if cls not in rolling_models:
            accumulated_samples[cls] = operating_batch[cls][unknowness_scores[cls]]
    return accumulated_samples
