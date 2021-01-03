import random
import numpy as np
import torch
import eval
from vast.tools import logger as vastlogger
from vast.clusteringAlgos import clustering
logger = vastlogger.get_logger()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def find_unknowness_probabilities(probabilities_for_train_set, unknowness_threshold=None):
    unknowness_scores = {}
    for k in set(probabilities_for_train_set.keys())-set(['classes_order']):
        unknowness_scores[k] = 1-torch.max(probabilities_for_train_set[k],dim=1).values
        if unknowness_threshold is not None:
            unknowness_scores[k] = unknowness_scores[k]>unknowness_threshold
    return unknowness_scores

@vastlogger.time_recorder
def mimic_incremental(args, current_batch,rolling_models, probabilities_for_train_set, batch_no):
    accumulated_samples = {}
    for c in current_batch:
        if c not in rolling_models:
            accumulated_samples[c] = current_batch[c]
    return accumulated_samples

@vastlogger.time_recorder
def learn_new_unknowns(args, operating_batch, class_already_enrolled, probabilities_for_train_set, batch_no):
    if len(class_already_enrolled)==0:
        return operating_batch
    unknowness_scores = find_unknowness_probabilities(probabilities_for_train_set[batch_no],
                                                      unknowness_threshold=args.unknowness_threshold)
    accumulated_samples = {}
    class_names = sorted(list(set(operating_batch.keys())))
    k = 0
    t = 0
    for cls in class_names:
        if cls not in class_already_enrolled:
            t += operating_batch[cls].shape[0]
            filtered_data = operating_batch[cls][unknowness_scores[cls]]
            if filtered_data.shape[0]==0: continue
            accumulated_samples[cls] = filtered_data
            k += accumulated_samples[cls].shape[0]
    logger.critical(f"Accumulated {k} samples from {t} unknowns")
    return accumulated_samples


@vastlogger.time_recorder
def learn_new_unknowns_UDA_Thresh(args, operating_batch, class_already_enrolled, probabilities_for_train_set, batch_no):
    if len(class_already_enrolled)==0:
        del probabilities_for_train_set[0]
        return operating_batch
    if batch_no==1:
        return learn_new_unknowns(args, operating_batch, class_already_enrolled, probabilities_for_train_set, batch_no)
    UDA, OCA, CCA, threshold_scores = eval.fixed_UDA_eval(probabilities_for_train_set, UDA_threshold=args.UDA_Threshold_for_training)
    logger.warning(f"Using the score threshold of {threshold_scores[-1]:.3f} to detect unknowns")
    unknowness_scores = find_unknowness_probabilities(probabilities_for_train_set[batch_no],
                                                      unknowness_threshold=threshold_scores[-1])
    accumulated_samples = {}
    class_names = sorted(list(set(operating_batch.keys())))
    k = 0
    t = 0
    for cls in class_names:
        # Done so that the classes that have already been enrolled are not learnt again
        if cls not in class_already_enrolled:
            accumulated_samples[cls] = operating_batch[cls][unknowness_scores[cls]]
            t += operating_batch[cls].shape[0]
            k += accumulated_samples[cls].shape[0]
    logger.critical(f"Accumulated {k} samples from {t} unknowns")
    return accumulated_samples


class OWL_on_a_budget():
    def __init__(self):
        self.stored_labels_gt=[]
        self.stored_labels_features=[]

    def __select_samples_for_labeling__(self, assignments, budget):
        """
        Randomly select from each cluster
        :param budget:
        :return: index of the samples to annotate
        """
        torch.manual_seed(0)
        logger.info("""
        Due to the random seed being set, the algorithm might request for labels for same samples in each pass. 
        But it won't effect performance as currently only first pass labels are used since class models are not updated in different passes
                    """)
        all_indices = torch.arange(assignments.shape[0])
        boolean_flags = torch.zeros(assignments.shape[0]).bool()
        per_sample_budget_ratio = budget/assignments.shape[0]
        for cluster_no in set(assignments.tolist())-{-1}:
            indxs_of_interest = all_indices[assignments==cluster_no]
            if indxs_of_interest.shape[0]<=5:
                continue
            budget_for_current_cluster = max(round(indxs_of_interest.shape[0] * per_sample_budget_ratio),1)
            i = indxs_of_interest[torch.randperm(indxs_of_interest.numel())[:budget_for_current_cluster]]
            boolean_flags[i] = True
        return boolean_flags

    def __assign_pseudo_labels_to_clusters__(self, assignments, features):
        pseudo_label_dict = {}
        stored_gt = np.array(self.stored_labels_gt)
        stored_features = np.array(self.stored_labels_features)
        assignments_for_stored_features = []
        for f in self.stored_labels_features:
            i = torch.where((features == torch.tensor(f).expand(features.shape[0], -1)).all(dim=1))[0]
            assignments_for_stored_features.append(assignments[i])
        assignments_for_stored_features = np.array(assignments_for_stored_features)
        for gt in set(stored_gt.tolist()):
            pseudo_label_dict[gt] = stored_features[stored_gt==gt].tolist()
        for a in set(assignments_for_stored_features.tolist()):
            gt = set(stored_gt[assignments_for_stored_features==a].tolist())
            if len(gt)==1:
                pseudo_label_dict[gt[0]].extend(features[assignments==a].tolist())
        for k in pseudo_label_dict:
            pseudo_label_dict[k] = torch.tensor(pseudo_label_dict[k]).double()
        return pseudo_label_dict

    @vastlogger.time_recorder
    def __OWL_on_a_budget__(self, args, operating_batch, OOD_model, probabilities_for_operating_batch):
        """
        :param operating_batch: This is the data whose subset will be used for training, its keys should never be used
        :param OOD_model:
        :param probabilities_for_operating_batch: These are probabilities of the operating batch, note keys should never be used
        :return:
        """

        # Filter knowns vs unknwons from the operating batch.
        # If we do not have any model trained for the knowns, consider all samples as unknowns.
        if len(OOD_model.keys()) == 0:
            accumulated_samples = operating_batch
            budget = args.initial_no_of_samples if args.initialization_batch_annotation_budget \
                                                   is None else args.initialization_batch_annotation_budget
        else:
            budget = args.annotation_budget
            unknowness_scores = find_unknowness_probabilities(probabilities_for_operating_batch,
                                                              unknowness_threshold=args.unknowness_threshold)
            accumulated_samples = {}
            class_names = sorted(list(set(operating_batch.keys())))
            for cls in class_names:
                accumulated_samples[cls] = operating_batch[cls][unknowness_scores[cls]]

        # Create a single tensor from all unknowns
        unknown_samples_features = []
        gt = []
        for cls in accumulated_samples:
            unknown_samples_features.append(accumulated_samples[cls])
            gt.extend([cls]*accumulated_samples[cls].shape[0])
        unknown_samples_features = torch.cat(unknown_samples_features)
        unknown_samples_features = unknown_samples_features.type(torch.FloatTensor)
        gt = np.array(gt)

        assert unknown_samples_features.shape[0]!=0, "No unknown samples detected"
        # Perform clustering on all unknown samples
        Clustering_Algo = getattr(clustering, args.Accumulator_clustering_Algo)
        centroids, assignments = Clustering_Algo(unknown_samples_features, K=min(unknown_samples_features.shape[0], 100),
                                                 verbose=False, distance_metric=args.distance_metric)
        # From the above clusters select the candidate samples that will be used for labeling
        annotation_candidate_indxs = self.__select_samples_for_labeling__(assignments, budget)
        # Hold the GT labels for the annotated samples.
        self.stored_labels_features.extend(unknown_samples_features[annotation_candidate_indxs,:].tolist())
        self.stored_labels_gt.extend(gt[annotation_candidate_indxs].tolist())
        # Using the annotations assign pseudo labels to clusters.
        pseudo_label_dict = self.__assign_pseudo_labels_to_clusters__(assignments, unknown_samples_features)
        # Return pseudo_label_dict
        return pseudo_label_dict

    def __call__(self, *args_passed, **kwargs):
        return self.__OWL_on_a_budget__(*args_passed, **kwargs)
