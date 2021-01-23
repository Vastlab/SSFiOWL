import torch
import numpy as np
from vast.tools import logger as vastlogger

logger = vastlogger.get_logger()

def calculate_CCA(results_for_all_batches):
    CCA = []
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        scores_order = np.array(results_for_all_batches[batch_no]['classes_order'])
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
        scores_order = np.array(results_for_all_batches[batch_no]['classes_order'])
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



def fixed_UDA_eval(results_for_all_batches, UDA_threshold=0.9):
    UDA = []
    OCA = []
    CCA = []
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        unknown_classes = (set(results_for_all_batches[batch_no].keys()) -
                           set(results_for_all_batches[batch_no]['classes_order']) -
                           {'classes_order'})
        unknown_classes = np.array(list(unknown_classes))
        current_batch_scores=[]
        current_batch_gt=[]
        current_batch_prediction=[]
        scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
        for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
            max_scores = torch.max(results_for_all_batches[batch_no][test_cls], dim=1)
            current_batch_scores.extend(max_scores.values.tolist())
            current_batch_prediction.extend(scores_order[max_scores.indices].tolist())
            current_batch_gt.extend([test_cls]*max_scores.values.shape[0])
        comb=list(zip(current_batch_scores,current_batch_prediction,current_batch_gt))
        comb.sort(reverse=True)
        current_batch_scores,current_batch_prediction,current_batch_gt=zip(*comb)
        current_batch_scores=np.array(current_batch_scores)
        current_batch_prediction=np.array(current_batch_prediction)
        current_batch_gt=np.array(current_batch_gt)

        UDA_correct = np.in1d(current_batch_gt, unknown_classes)
        UDA_correct = np.cumsum(UDA_correct)
        UDA_correct = UDA_correct/UDA_correct[-1]
        OCA_correct = np.in1d(current_batch_gt, unknown_classes)
        OCA_correct[~OCA_correct] = current_batch_gt[~OCA_correct]==current_batch_prediction[~OCA_correct]
        OCA_correct = np.cumsum(OCA_correct)
        OCA_correct = OCA_correct/OCA_correct.shape[0]
        temp = ~np.in1d(current_batch_gt, unknown_classes)
        CCA_correct = np.zeros(current_batch_gt.shape[0], dtype='bool')
        CCA_correct[temp] = current_batch_gt[temp]==current_batch_prediction[temp]
        CCA_correct = np.cumsum(CCA_correct)
        CCA_correct = CCA_correct/sum(temp)
        UDA.append(UDA_correct[UDA_correct<=UDA_threshold][-1]*100.)
        OCA.append(OCA_correct[UDA_correct<=UDA_threshold][-1]*100.)
        CCA.append(CCA_correct[UDA_correct<=UDA_threshold][-1]*100.)
        logger.critical(f"Batch {batch_no} : UDA {UDA[-1]:.2f}\t OCA {OCA[-1]:.2f}\t CCA {CCA[-1]:.2f}")
    logger.critical(f"Average Unknowness Accuracy : {np.mean(UDA):.2f} OCA {np.mean(OCA):.2f} CCA {np.mean(CCA):.2f}")
    return UDA, OCA, CCA
