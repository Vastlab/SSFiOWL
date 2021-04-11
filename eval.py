import torch
import numpy as np
import termtables as tt
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


def eval_data_prep(current_batch_scores, current_batch_prediction, current_batch_gt, unknown_classes):
    comb = list(zip(current_batch_scores, current_batch_prediction, current_batch_gt))
    comb.sort(reverse=True)
    current_batch_scores, current_batch_prediction, current_batch_gt = zip(*comb)
    # Detections are now sorted in decreasing order of probability scores
    current_batch_scores = np.array(current_batch_scores)
    current_batch_prediction = np.array(current_batch_prediction)
    current_batch_gt = np.array(current_batch_gt)
    known_flag = ~np.in1d(current_batch_gt, unknown_classes)
    UDA_correct = ~known_flag
    UDA_correct = np.cumsum(UDA_correct)
    UDA_correct = UDA_correct[-1] - UDA_correct
    UDA_correct = UDA_correct / max(UDA_correct[0], 1)
    OCA_correct = np.zeros(current_batch_gt.shape[0], dtype='bool')
    OCA_correct[known_flag] = current_batch_gt[known_flag] == current_batch_prediction[known_flag]
    OCA_correct = np.cumsum(OCA_correct)
    OCA_correct = OCA_correct / OCA_correct.shape[0]
    CCA_correct = np.zeros(current_batch_gt.shape[0], dtype='bool')
    CCA_correct[known_flag] = current_batch_gt[known_flag] == current_batch_prediction[known_flag]
    CCA_correct = np.cumsum(CCA_correct)
    CCA_correct = CCA_correct / sum(known_flag)
    # Threshold tie breaking
    current_batch_scores, indx = np.unique(current_batch_scores, return_index=True)
    current_batch_scores, indx = current_batch_scores[::-1], indx[::-1]
    indx = indx[1:]-1
    indx = np.array(indx.tolist() + [CCA_correct.shape[0] - 1])
    return current_batch_scores, CCA_correct[indx], UDA_correct[indx], OCA_correct[indx]



def calculate_CCA_on_thresh(results_for_all_batches, threshold=0.):
    UDA = []
    OCA = []
    CCA = []
    table_data = []
    logger.critical(f"Evaluating CCA at probability threshold of {threshold}")
    for batch_no in sorted(results_for_all_batches.keys()):
        unknown_classes = np.array([])
        current_batch_scores=[]
        current_batch_gt=[]
        current_batch_prediction=[]
        scores_order = np.array(results_for_all_batches[batch_no]['classes_order'])
        for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
            max_scores = torch.max(results_for_all_batches[batch_no][test_cls], dim=1)
            current_batch_scores.extend(max_scores.values.tolist())
            current_batch_prediction.extend(scores_order[max_scores.indices].tolist())
            current_batch_gt.extend([test_cls]*max_scores.values.shape[0])
        current_batch_scores, CCA_correct, UDA_correct, OCA_correct = eval_data_prep(current_batch_scores,
                                                                                     current_batch_prediction,
                                                                                     current_batch_gt, unknown_classes)
        UDA.append(UDA_correct[current_batch_scores>=threshold][-1]*100.)
        OCA.append(OCA_correct[current_batch_scores>=threshold][-1]*100.)
        CCA.append(CCA_correct[current_batch_scores>=threshold][-1]*100.)
        table_data.append((batch_no, f"{UDA[-1]:.2f}", f"{OCA[-1]:.2f}", f"{CCA[-1]:.2f}"))
    table_data.append(("Average", f"{np.mean(UDA):.2f}", f"{np.mean(OCA):.2f}", f"{np.mean(CCA):.2f}"))
    table_header = ["Batch No", "UDA", "OCA", "CCA"]
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=tt.styles.rounded_thick,
                                  alignment="cccc",
                                  padding=(0, 1))
    logger.error("\n" + table_data_str)
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=" &             ",
                                  alignment="cccc",
                                  padding=(0, 1))
    logger.warning("\n\nTable in latex format\n\n"+table_data_str+"\n\n")
    return UDA, OCA, CCA



def fixed_probability_score(results_for_all_batches, unknowness_threshold=0.5):
    UDA = []
    OCA = []
    CCA = []
    table_data = []
    threshold_scores = []
    logger.critical(f"Evaluation at fixed probability score of {unknowness_threshold}")
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        unknown_classes = (set(results_for_all_batches[batch_no].keys()) -
                           set(results_for_all_batches[batch_no]['classes_order']) -
                           {'classes_order'})
        unknown_classes = np.array(list(unknown_classes))
        current_batch_scores=[]
        current_batch_gt=[]
        current_batch_prediction=[]
        scores_order = np.array(results_for_all_batches[batch_no]['classes_order'])
        for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
            max_scores = torch.max(results_for_all_batches[batch_no][test_cls], dim=1)
            current_batch_scores.extend(max_scores.values.tolist())
            current_batch_prediction.extend(scores_order[max_scores.indices].tolist())
            current_batch_gt.extend([test_cls]*max_scores.values.shape[0])
        current_batch_scores, CCA_correct, UDA_correct, OCA_correct = eval_data_prep(current_batch_scores,
                                                                                     current_batch_prediction,
                                                                                     current_batch_gt, unknown_classes)
        UDA.append(UDA_correct[current_batch_scores>=unknowness_threshold][-1]*100.)
        OCA.append(OCA_correct[current_batch_scores>=unknowness_threshold][-1]*100.)
        CCA.append(CCA_correct[current_batch_scores>=unknowness_threshold][-1]*100.)
        threshold_scores.append(current_batch_scores[current_batch_scores>=unknowness_threshold][-1])
        table_data.append((batch_no, f"{threshold_scores[-1]:.3f}", f"{UDA[-1]:.2f}", f"{OCA[-1]:.2f}", f"{CCA[-1]:.2f}"))
    table_data.append(("Average", "", f"{np.mean(UDA):.2f}", f"{np.mean(OCA):.2f}", f"{np.mean(CCA):.2f}"))
    table_header = ["Batch No", "Score", "UDA", "OCA", "CCA"]
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=tt.styles.rounded_thick,
                                  alignment="ccccc",
                                  padding=(0, 1))
    logger.error("\n" + table_data_str)
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=" &             ",
                                  alignment="ccccc",
                                  padding=(0, 1))
    logger.warning("\n\nTable in latex format\n\n"+table_data_str+"\n\n")
    return UDA, OCA, CCA


def fixed_UDA_eval(results_for_all_batches, UDA_threshold=0.9):
    UDA = []
    OCA = []
    CCA = []
    threshold_scores = []
    table_data = []
    new_classes_per_batch = []
    no_of_batches_considered = len(results_for_all_batches.keys())-1
    all_classes_seen_till_batch=set()
    for batch_no in sorted(results_for_all_batches.keys())[:-1]:
        new_classes_per_batch.append(set(results_for_all_batches[batch_no]['classes_order'])-all_classes_seen_till_batch)
        all_classes_seen_till_batch=all_classes_seen_till_batch.union(new_classes_per_batch[-1])
        unknown_classes = (set(results_for_all_batches[batch_no].keys()) -
                           set(results_for_all_batches[batch_no]['classes_order']) -
                           {'classes_order'})
        unknown_classes = np.array(list(unknown_classes))
        current_batch_scores=[]
        current_batch_gt=[]
        current_batch_prediction=[]
        scores_order = np.array(results_for_all_batches[batch_no]['classes_order'])
        for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
            max_scores = torch.max(results_for_all_batches[batch_no][test_cls], dim=1)
            current_batch_scores.extend(max_scores.values.tolist())
            current_batch_prediction.extend(scores_order[max_scores.indices].tolist())
            current_batch_gt.extend([test_cls]*max_scores.values.shape[0])
        current_batch_scores_, CCA_correct, UDA_correct, OCA_correct = eval_data_prep(current_batch_scores,
                                                                                     current_batch_prediction,
                                                                                     current_batch_gt, unknown_classes)
        UDA.append(UDA_correct[UDA_correct>=UDA_threshold][-1]*100.)
        OCA.append(OCA_correct[UDA_correct>=UDA_threshold][-1]*100.)
        CCA.append(CCA_correct[UDA_correct>=UDA_threshold][-1]*100.)
        threshold_scores.append(current_batch_scores_[UDA_correct>=UDA_threshold][-1])

        per_incremental_batch_CCA = []
        for a in new_classes_per_batch:
            classes_to_consider_as_unknowns = set(unknown_classes).union(all_classes_seen_till_batch) - a
            classes_to_consider_as_unknowns = np.array(list(classes_to_consider_as_unknowns))
            current_batch_scores_, CCA_correct, _, _ = eval_data_prep(current_batch_scores,
                                                                      current_batch_prediction,
                                                                      current_batch_gt,
                                                                      classes_to_consider_as_unknowns)
            assert CCA_correct.shape[0]==UDA_correct.shape[0],\
                "Threshold tie breaking might be causing an issue"
            if len(CCA_correct[UDA_correct >= UDA_threshold])>0:
                per_incremental_batch_CCA.append(f"{CCA_correct[UDA_correct >= UDA_threshold][-1] * 100.:.2f}")
            else:
                per_incremental_batch_CCA.append("0.0")
        for k in range(no_of_batches_considered-len(new_classes_per_batch)):
            per_incremental_batch_CCA.append("")

        table_row = [batch_no, f"{threshold_scores[-1]:.3f}", f"{UDA[-1]:.2f}", f"{OCA[-1]:.2f}", f"{CCA[-1]:.2f}"]
        table_row.extend(per_incremental_batch_CCA)
        table_data.append(table_row)
    table_row = ["Average", "", f"{np.mean(UDA):.2f}", f"{np.mean(OCA):.2f}", f"{np.mean(CCA):.2f}"]
    table_row.extend([""]*no_of_batches_considered)
    table_data.append(table_row)
    table_header = ["Batch No", "Score", "UDA", "OCA", "CCA"]
    table_header.extend([f"B.No.{i}" for i in range(no_of_batches_considered)])
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=tt.styles.rounded_thick,
                                  alignment="ccccc"+''.join(["c"]*no_of_batches_considered),
                                  padding=(0, 1))
    logger.error("\n"+table_data_str)
    table_data_str = tt.to_string(table_data,
                                  header=table_header,
                                  style=" &             ",
                                  alignment="ccccc"+''.join(["c"]*no_of_batches_considered),
                                  padding=(0, 1))
    logger.warning("\n\nTable in latex format\n\n"+table_data_str+"\n\n")
    return UDA, OCA, CCA, threshold_scores
