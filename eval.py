import torch
import numpy as np
from utile.tools import logger as utilslogger

logger = utilslogger.get_logger()

def calculate_CCA(results_for_all_batches):
    CCA = []
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
        correct = 0.
        total = 0.
        for test_cls in scores_order.tolist():
            scores = results_for_all_batches[batch_no][test_cls]
            total += scores.shape[0]
            max_indx = torch.argmax(scores, dim=1)
            correct += sum(scores_order[max_indx] == test_cls)
        CCA.append((correct / total) * 100.)
        logger.critical(f"Accuracy on Batch {batch_no} : {CCA[-1]}")
    logger.critical(f"Average Closed Set Classification Accuracy : {np.mean(CCA)}")
    return CCA

def calculate_UDA_OCA(results_for_all_batches, unknowness_threshold=0.5):
    UDA = []
    OCA = []
    batches = []
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        unknown_classes = (set(results_for_all_batches[batch_no].keys()) -
                           set(results_for_all_batches[batch_no]['classes_order']) -
                           {'classes_order'})
        scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
        UDA_correct = 0.
        UDA_total = 0.
        OCA_correct = 0.
        total = 0.
        batches.append(results_for_all_batches[batch_no][list(results_for_all_batches[batch_no].keys())[0]].shape[1])
        for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
            scores = results_for_all_batches[batch_no][test_cls]
            total += scores.shape[0]
            max_scores = torch.max(scores, dim=1)
            unknowness_scores = 1 - max_scores.values
            predicted_as_unknowns = sum(unknowness_scores > unknowness_threshold)
            if test_cls in unknown_classes:
                UDA_correct += predicted_as_unknowns
                OCA_correct += predicted_as_unknowns
                UDA_total += scores.shape[0]
            else:
                temp = np.array([scores_order[max_scores.indices[unknowness_scores <= unknowness_threshold]]])
                if len(temp.shape) > 1:
                    temp = np.squeeze(temp)
                OCA_correct += sum(temp == test_cls)
        UDA.append((UDA_correct / max(UDA_total, 1)) * 100.)
        OCA.append((OCA_correct / total) * 100.)
        logger.critical(f"Unknowness detection accuracy on Batch {batch_no} : {UDA[-1]}  OCA {OCA[-1]}")
    logger.critical(f"Average Unknowness Accuracy : {np.mean(UDA)} OCA {np.mean(OCA)}")
    return UDA, OCA, batches
