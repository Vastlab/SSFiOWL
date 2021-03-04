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
    parser.add_argument('--OOD_Algo', default='OpenMax', type=str, choices=['OpenMax','EVM','MultiModalOpenMax'],
                        help='Name of the Out of Distribution detection algorithm\ndefault: %(default)s')

    parser = data_prep.params(parser)

    parser.add_argument('--accumulation_algo', default='learn_new_unknowns', type=str,
                        help='Name of the accumulation algorithm to use\ndefault: %(default)s',
                        choices=['mimic_incremental','learn_new_unknowns','update_existing_learn_new'])
    parser.add_argument("--unknowness_threshold", type=float, default=0.5,
                        help="unknowness probability score above which a sample is considered as unknown\n"
                             "Note: Cannot be a list because this varies results at each operational batch"
                             "default: %(default)s")
    parser.add_argument("--UDA_Threshold", nargs="+", type=float, default=[0.7, 0.8, 0.9, 0.95, 1.0],
                                   help="tail size to use\ndefault: %(default)s")

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

    known_args, unknown_args = parser.parse_known_args()

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs the open world experiments from the paper"
                                                          "i.e. Table 3")
    parser, _ = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
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

    results_for_all_batches = {}
    completed_q = mp.Queue()
    list_of_all_batch_nos = set(batch_nos.tolist())
    net_ops_obj = network_operations.netowrk(num_classes=0)
    for batch in list_of_all_batch_nos:
        logger.info(f"Preparing batch {batch} from training data (initialization/operational)")
        current_batch = get_current_batch(classes, features, batch_nos, batch, images)

        logger.info(f"Processing batch {batch}/{len(list_of_all_batch_nos)}")
        probabilities_for_train_set={}
        if len(net_ops_obj.cls_names)>0:
            logger.info(f"Getting probabilities for the current operational batch")
            probabilities_for_train_set = net_ops_obj.inference(validation_data=current_batch)
            probabilities_for_train_set['classes_order'] = net_ops_obj.cls_names

        # Accumulate all unknown samples
        accumulated_samples = accumulation_algo(args, current_batch, net_ops_obj.cls_names, probabilities_for_train_set)

        exemplars_to_add={}
        # Add exemplars
        if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
            exemplars_to_add = exemplar_selection.random_selector(features, net_ops_obj.cls_names,
                                                                  no_of_exemplars=args.no_of_exemplars)
            accumulated_samples.update(exemplars_to_add)
        # Add all negative samples
        if args.all_samples and batch!=0:
            exemplars_to_add = exemplar_selection.add_all_negatives(features, net_ops_obj.cls_names)
            current_batch.update(exemplars_to_add)

        # Run enrollment for unknown samples probabilities_for_train_set
        no_of_classes_to_enroll = len(accumulated_samples) - len(exemplars_to_add)
        logger.info(f"{f' Enrolling {no_of_classes_to_enroll} new classes with {len(exemplars_to_add)} exemplar batches '.center(90, '#')}")
        net_ops_obj.training(training_data=accumulated_samples,
                             lr=1e-2 if batch==0 else 1e-3,
                             epochs=400 if batch==0 else 300)

        logger.info(f"Preparing validation data")
        current_batch = get_current_batch(val_classes, val_features, val_batch_nos, 0, val_images,
                                          classes_to_fetch=set(classes[batch_nos<=min(batch+1,max(batch_nos))].tolist()))

        logger.info(f"Running on validation data")
        results_for_all_batches[batch] = net_ops_obj.inference(validation_data=current_batch)
        results_for_all_batches[batch]['classes_order'] = net_ops_obj.cls_names

    dir_name = f"OpenWorld_Learning/InitialClasses-{args.initialization_classes}_TotalClasses-{args.total_no_of_classes}" \
               f"_NewClassesPerBatch-{args.new_classes_per_batch}"
    if args.OOD_Algo == 'EVM':
        file_name = f"{args.distance_metric}_{args.OOD_Algo}Params-{args.tailsize}_{args.cover_threshold}_{args.distance_multiplier}"
    else:
        file_name = f"{args.distance_metric}_{args.OOD_Algo}Params-{args.tailsize}_{args.distance_multiplier}"
    if args.all_samples:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/all_samples/")
    else:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/no_of_exemplars_{args.no_of_exemplars}/")
    file_path.mkdir(parents=True, exist_ok=True)
    logger.critical(f"Saving to path {file_path}")
    pickle.dump(results_for_all_batches, open(f"{file_path}/{args.OOD_Algo}_{file_name}.pkl", "wb"))
    UDA, OCA, CCA = eval.fixed_probability_score(results_for_all_batches, unknowness_threshold=args.unknowness_threshold)
    logger.critical(f"For Tabeling")
    logger.critical(f"Thresholding on scores {np.mean(UDA):.2f} & {np.mean(OCA):.2f} & {np.mean(CCA):.2f}")
    for UDA_threshold in args.UDA_Threshold:
        UDA_on_fixed_UDA, OCA_on_fixed_UDA, CCA_on_fixed_UDA = eval.fixed_UDA_eval(results_for_all_batches,
                                                                                      UDA_threshold=UDA_threshold)
        logger.critical(f"For fixed UDA of {UDA_threshold}\t {np.mean(UDA_on_fixed_UDA):.2f} "
                        f"& {np.mean(OCA_on_fixed_UDA):.2f} & {np.mean(CCA_on_fixed_UDA):.2f}")