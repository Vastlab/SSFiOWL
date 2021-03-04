import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
import pathlib
import itertools
import protocols
import exemplar_selection
import data_prep
import network_operations
import viz
from vast import opensetAlgos
from vast.tools import logger as vastlogger
import eval

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
    parser.add_argument('--OOD_Algo', default='EVM', type=str, choices=['OpenMax','EVM','MultiModalOpenMax'],
                        help='Name of the openset detection algorithm\ndefault: %(default)s')

    parser = data_prep.params(parser)

    protocol_params = parser.add_argument_group('Protocol params')
    protocol_params.add_argument("--total_no_of_classes", type=int, default=100,
                                 help="Total no of classes\ndefault: %(default)s")
    protocol_params.add_argument("--initialization_classes", type=int, default=50,
                                 help="No of classes in first batch\ndefault: %(default)s")
    protocol_params.add_argument("--new_classes_per_batch", type=int, default=[1,2,5,10], nargs="+",
                                 help="No of new classes added per batch\ndefault: %(default)s")

    known_args, unknown_args = parser.parse_known_args()

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs experiments for incremental learning " 
                                                          "i.e. Table 1 and 2 from the Paper")
    parser, _ = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True
    if args.debug:
        args.verbose = 0
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)
    all_new_classes_per_batch = args.new_classes_per_batch

    all_grid_search_results = []
    per_class_best_results = []
    tabular_results = []

    org_dm, org_ct = args.distance_multiplier, args.cover_threshold

    for exp_no, new_classes_per_batch in enumerate(all_new_classes_per_batch):
        args.new_classes_per_batch = new_classes_per_batch

        # Get the protocols
        batch_nos, images, classes = protocols.ImageNetIncremental(initial_no_of_classes=args.initialization_classes,
                                                                   new_classes_per_batch=args.new_classes_per_batch,
                                                                   total_classes=args.total_no_of_classes)
        val_batch_nos, val_images, val_classes = protocols.ImageNetIncremental(files_to_add = ['imagenet_1000_val'],
                                                                               initial_no_of_classes=args.initialization_classes,
                                                                               new_classes_per_batch=args.new_classes_per_batch,
                                                                               total_classes=args.total_no_of_classes)

        # Read all Features
        if exp_no == 0:
            args.feature_files = args.training_feature_files
            features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(classes.tolist())))
            args.feature_files = args.validation_feature_files
            val_features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(val_classes.tolist())))

        net_ops_obj = network_operations.netowrk(num_classes=0)

        results_for_all_batches = {}
        completed_q = mp.Queue()
        for batch in set(batch_nos.tolist()):
            logger.info(f"Preparing training batch {batch}")
            current_batch = {}
            # Add exemplars
            if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
                current_batch.update(exemplar_selection.random_selector(features, net_ops_obj.cls_names, no_of_exemplars=args.no_of_exemplars))
            # Add all negative samples
            if args.all_samples and batch!=0:
                current_batch.update(exemplar_selection.add_all_negatives(features, net_ops_obj.cls_names))

            for cls in sorted(set(classes[batch_nos==batch].tolist())-set(net_ops_obj.cls_names)):
                indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes==cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:,None].expand(-1, features[cls]['features'].shape[1])
                current_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
            logger.info(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

            no_of_classes_to_process = len(set(classes[batch_nos==batch].tolist()))
            net_ops_obj.training(training_data=current_batch ,lr=1e-2)

            logger.info(f"Preparing validation data")
            current_batch = {}
            for cls in sorted(set(val_classes[val_batch_nos==batch].tolist())):
                indx_of_interest = np.where(np.in1d(val_features[cls]['images'], val_images[(val_batch_nos == batch) & (val_classes==cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:,None].expand(-1, val_features[cls]['features'].shape[1])
                current_batch[cls] = val_features[cls]['features'].gather(0, indx_of_interest)
            logger.info(f"Running on validation data")

            results_for_all_batches[batch] = net_ops_obj.inference(validation_data=current_batch)
            results_for_all_batches[batch]['classes_order'] = net_ops_obj.cls_names

        args.output_dir=pathlib.Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.critical(f"Results for {new_classes_per_batch} new classes/batch DM {args.distance_multiplier}"
                        f" CT {args.cover_threshold}")

        UDA, OCA, CCA = eval.calculate_CCA_on_thresh(results_for_all_batches, threshold=0.)
        all_grid_search_results.append((f"{new_classes_per_batch:03d}", f"{np.mean(CCA):06.2f}",
                                        f"{args.distance_multiplier[0]}", f"{args.cover_threshold[0]}"))

        all_grid_search_results = sorted(all_grid_search_results)
        per_class_best_results.append(' '.join(all_grid_search_results[-1]))
        tabular_results.append(all_grid_search_results[-1][1])
        logger.critical("Grid Search Results:\n"+'\n'.join((' '.join(_) for _ in all_grid_search_results)))
    logger.critical("Best Results:\n"+'\n'.join(per_class_best_results))

    if len(all_new_classes_per_batch)>1:
        logger.critical(' & '.join(tabular_results))