import argparse
import pickle
import pathlib
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.multiprocessing as mp
import protocols
import data_prep
import common_operations
import network_operations
import exemplar_selection
import eval
import accumulation_algos
from vast import opensetAlgos
from vast.tools import logger as vastlogger

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     add_help=False, usage=argparse.SUPPRESS)

    parser.add_argument('-v', '--verbose', help="To decrease verbosity increase", action='count', default=0)
    parser.add_argument("--debug", action="store_true", default=False, help="debugging flag\ndefault: %(default)s")
    parser.add_argument("--no_multiprocessing", action="store_true", default=False,
                        help="Use for debugging or running on single GPU\ndefault: %(default)s")
    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing\ndefault: %(default)s')
    parser.add_argument("--no_of_exemplars", type=int, default=0,
                        help="No of exemplars used during incremental step\ndefault: %(default)s")
    parser.add_argument("--all_samples", action="store_true", default=False,
                        help="Enroll new classes considering all previously encountered samples\ndefault: %(default)s")

    parser.add_argument("--output_dir", type=str, default='/scratch/adhamija/results/', help="Results directory")

    parser.add_argument('--OOD_Algo', default='MLP', type=str, choices=['OpenMax','EVM','MultiModalOpenMax','MLP'],
                        help='Name of the approach')

    parser = data_prep.params(parser)

    parser.add_argument('--accumulation_algo', default='learn_new_unknowns', type=str,
                        help='Name of the accumulation algorithm to use\ndefault: %(default)s',
                        choices=['mimic_incremental','learn_new_unknowns','update_existing_learn_new',
                                 'learn_new_unknowns_UDA_Thresh','OWL_on_a_budget'])
    parser.add_argument("--UDA_Threshold_for_training", type=float, default=0.7,
                        help="UDA threshold used to decide unknowness threshold for enrolling samples from "
                             "next operational batch\ndefault: %(default)s")
    parser.add_argument("--unknowness_threshold", type=float, default=0.5,
                        help="unknowness probability score above which a sample is considered as unknown\n"
                             "Note: Cannot be a list because this varies results at each operational batch"
                             "default: %(default)s")
    parser.add_argument("--UDA_Threshold", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                                   help="UDA Threshold for evaluation\ndefault: %(default)s")

    protocol_params = parser.add_argument_group('Protocol params')
    protocol_params.add_argument("--total_no_of_classes", type=int, default=100,
                                 help="Total no of classes\ndefault: %(default)s")
    protocol_params.add_argument("--initialization_classes", type=int, default=50,
                                 help="No of classes in first batch\ndefault: %(default)s")
    protocol_params.add_argument("--new_classes_per_batch", type=int, default=5,
                                 help="No of new classes added per batch\ndefault: %(default)s")
    protocol_params.add_argument("--known_sample_per_batch", type=int, default=2500,
                                 help="Samples belonging to known classes in every incremental batch\ndefault: %(default)s")
    protocol_params.add_argument("--unknown_sample_per_batch", type=int, default=2500,
                                 help="Samples belonging to unknown classes in every incremental batch\ndefault: %(default)s")
    protocol_params.add_argument("--initial_no_of_samples", type=int, default=15000,
                                 help="Number of samples in the first/initialization batch\ndefault: %(default)s")

    parser.add_argument('--Accumulator_clustering_Algo', default='finch', type=str,
                        choices=['KMeans','dbscan','finch'],
                        help='Clustering algorithm used for the accumulation algorithm default: %(default)s')
    parser.add_argument("--annotation_budget", type=int, default=0,
                        help="Annotation Budget\ndefault: %(default)s")
    parser.add_argument("--initialization_batch_annotation_budget", type=int, default=None,
                        help="Initialization Batch Annotation Budget\ndefault: %(default)s")

    known_args, unknown_args = parser.parse_known_args()

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs the open world experiments from the paper"
                                                          "i.e. Table 3")
    if known_args.OOD_Algo!="MLP":
        parser, _ = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    else:
        MLP_params_parser = params_parser.add_argument_group(title="MLP", description="MLP params")
        MLP_params_parser.add_argument("--lr", nargs="+", type=float, default=[1e-2, 1e-3],
                                       help="Learning rate to use at various learning batches."
                                            "If two lr are provided they correspond to 1st and rest of the batches.")
        MLP_params_parser.add_argument("--epochs", nargs="+", type=int, default=[400, 300],
                                       help="Number of epochs to train for each batch."
                                            "If two numbers are provided they correspond to the 1st and rest")
        parser = params_parser
    args = parser.parse_args()
    return args


def get_current_batch(classes, features, batch_nos, batch, images, classes_to_fetch=None):
    if classes_to_fetch is None:
        classes_to_fetch = sorted(set(classes[batch_nos == batch].tolist()))
    current_batch = {}
    for cls in classes_to_fetch:
        indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes == cls)]))[0]
        indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
        indx_of_interest = indx_of_interest[:, None].expand(-1, features[cls]['features'].shape[1])
        current_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
    return current_batch

if __name__ == "__main__":
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True
    if args.debug:
        args.verbose = 0
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)

    accumulation_algo = getattr(accumulation_algos, args.accumulation_algo)

    if args.accumulation_algo == "OWL_on_a_budget":
        accumulation_algo = accumulation_algos.OWL_on_a_budget()
    else:
        accumulation_algo = getattr(accumulation_algos, args.accumulation_algo)

    if args.accumulation_algo == "OWL_on_a_budget":
        accumulation_algo = accumulation_algos.OWL_on_a_budget()
    else:
        accumulation_algo = getattr(accumulation_algos, args.accumulation_algo)

    # Get the operational protocols
    batch_nos, images, classes = protocols.open_world_protocol(initial_no_of_classes=args.initialization_classes,
                                                               new_classes_per_batch=args.new_classes_per_batch,
                                                               initial_batch_size=args.initial_no_of_samples,
                                                               known_sample_per_batch=args.known_sample_per_batch,
                                                               unknown_sample_per_batch=args.unknown_sample_per_batch,
                                                               total_classes=args.total_no_of_classes)
    val_batch_nos, val_images, val_classes = protocols.OpenWorldValidation(files_to_add = ['imagenet_1000_val'],
                                                                           classes=set(classes.tolist()))

    # Read all Features
    args.feature_files = args.training_feature_files
    features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(classes.tolist())))
    args.feature_files = args.validation_feature_files
    val_features = data_prep.prep_all_features_parallel(args)

    completed_q = mp.Queue()
    event = mp.Event()
    list_of_all_batch_nos = set(batch_nos.tolist())
    results_for_all_batches = {}
    models_across_batches = {}
    probabilities_for_train_set = {}
    stored_exemplars = {}
    for batch in list_of_all_batch_nos:
        logger.info(f"Preparing batch {batch} from training data (initialization/operational)")
        current_batch = get_current_batch(classes, features, batch_nos, batch, images)

        logger.info(f"Processing batch {batch}/{len(list_of_all_batch_nos)}")
        probabilities_for_train_set[batch]={}
        if len(models_across_batches)>0:
            logger.info(f"Getting probabilities for the current operational batch")
            common_operations.call_specific_approach(0, batch, args, current_batch, completed_q,
                                                     event=event, models=models_across_batches)
            probabilities_for_train_set[batch].update(common_operations.convert_q_to_dict(args, completed_q,
                                                                                      None, event=event))
            probabilities_for_train_set[batch]['classes_order'] = [*models_across_batches]

        # Accumulate all unknown samples
        accumulated_samples = accumulation_algo(args, current_batch, [*models_across_batches],
                                                probabilities_for_train_set, batch)

        # Add exemplars
        # Based on a set number of exemplars
        if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
            exemplars_to_add = exemplar_selection.random_selector(current_batch, [*models_across_batches],
                                                                  no_of_exemplars=args.no_of_exemplars)
            for e in exemplars_to_add:
                if e not in stored_exemplars:
                    stored_exemplars[e] = exemplars_to_add[e]
            accumulated_samples.update(stored_exemplars)
        # Add all negative samples
        if args.all_samples and batch!=0:
            logger.warning("Taking all samples as exemplars")
            exemplars_to_add = exemplar_selection.add_all_negatives(current_batch, [*models_across_batches])
            for e in exemplars_to_add:
                if e not in stored_exemplars:
                    stored_exemplars[e] = exemplars_to_add[e]
            accumulated_samples.update(stored_exemplars)

        # Run enrollment for unknown samples
        no_of_classes_to_enroll = len(accumulated_samples) - len(stored_exemplars)
        logger.info(f"{f' Enrolling {no_of_classes_to_enroll} new classes with {len(stored_exemplars)} exemplar batches '.center(90, '#')}")

        common_operations.call_specific_approach(0, batch, args, accumulated_samples, completed_q, event=event)  #current_batch
        model = common_operations.convert_q_to_dict(args, completed_q, None, event=event)

        logger.info(f"Preparing validation data")
        current_validation_batch = get_current_batch(val_classes, val_features, val_batch_nos, 0, val_images,
                                          classes_to_fetch=set(classes[batch_nos<=min(batch+1,max(batch_nos))].tolist()))

        results_for_all_batches[batch] = {}
        if args.OOD_Algo == "MLP":
            models_across_batches = dict.fromkeys(model.cls_names)
            results_for_all_batches[batch]['classes_order'] = model.cls_names
        else:
            models_across_batches.update(model)
            results_for_all_batches[batch]['classes_order'] = sorted([*models_across_batches])

        logger.info(f"Running on validation data")
        common_operations.call_specific_approach(0, batch, args, current_validation_batch, completed_q,
                                                 event=event, models=models_across_batches)
        results_for_all_batches[batch].update(common_operations.convert_q_to_dict(args, completed_q,
                                                                                  None, event=event))

    UDA, OCA, CCA = eval.fixed_probability_score(results_for_all_batches, unknowness_threshold=args.unknowness_threshold)
    logger.critical(f"For Tabeling")
    logger.critical(f"Thresholding on scores {np.mean(UDA):.2f} & {np.mean(OCA):.2f} & {np.mean(CCA):.2f}")
    for UDA_threshold in args.UDA_Threshold:
        UDA_on_fixed_UDA, OCA_on_fixed_UDA, CCA_on_fixed_UDA, _ = eval.fixed_UDA_eval(results_for_all_batches,
                                                                                      UDA_threshold=UDA_threshold)
        logger.critical(f"For fixed UDA of {UDA_threshold}\t {np.mean(UDA_on_fixed_UDA):.2f} "
                        f"& {np.mean(OCA_on_fixed_UDA):.2f} & {np.mean(CCA_on_fixed_UDA):.2f}")
