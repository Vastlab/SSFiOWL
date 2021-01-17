import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.multiprocessing as mp
import pathlib
import itertools
import protocols
import exemplar_selection
import data_prep
import common_operations
import viz
from utile import opensetAlgos
from utile.tools import logger as utilslogger


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
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True
    if args.debug:
        args.verbose = 0
    logger = utilslogger.setup_logger(level=args.verbose, output=args.output_dir)
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

        if len(all_new_classes_per_batch) == len(org_dm) == len(org_ct):
            combinations_to_try = ((org_dm[exp_no], org_ct[exp_no]),)
        else:
            combinations_to_try = itertools.product(org_dm, org_ct)

        event = mp.Event()
        for dm, ct in combinations_to_try:
            args.distance_multiplier, args.cover_threshold = [dm], [ct]

            rolling_models = {}
            results_for_all_batches = {}
            completed_q = mp.Queue()
            for batch in set(batch_nos.tolist()):
                logger.info(f"Preparing training batch {batch}")
                current_batch = {}
                # Add exemplars
                if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
                    current_batch.update(exemplar_selection.random_selector(features, rolling_models, no_of_exemplars=args.no_of_exemplars))
                # Add all negative samples
                if args.all_samples and batch!=0:
                    current_batch.update(exemplar_selection.add_all_negatives(features, rolling_models))

                for cls in sorted(set(classes[batch_nos==batch].tolist())-set(rolling_models.keys())):
                    indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes==cls)]))[0]
                    indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                    indx_of_interest = indx_of_interest[:,None].expand(-1, features[cls]['features'].shape[1])
                    current_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
                logger.info(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

                event.clear()
                no_of_classes_to_process = len(set(classes[batch_nos==batch].tolist())-set(rolling_models.keys()))
                if args.no_multiprocessing or no_of_classes_to_process==1:
                    args.world_size = 1
                    common_operations.call_specific_approach(0, args, current_batch, completed_q, event)
                    p=None
                    models = common_operations.convert_q_to_dict(args, completed_q, p, event)
                else:
                    args.world_size = min(no_of_classes_to_process, torch.cuda.device_count())
                    p = mp.spawn(common_operations.call_specific_approach,
                                 args=(args, current_batch, completed_q, event),
                                 nprocs=args.world_size,
                                 join=False)
                    models = common_operations.convert_q_to_dict(args, completed_q, p, event)

                logger.info(f"Preparing validation data")
                rolling_models.update(models)
                current_batch = {}
                for cls in sorted(set(val_classes[val_batch_nos==batch].tolist())):
                    indx_of_interest = np.where(np.in1d(val_features[cls]['images'], val_images[(val_batch_nos == batch) & (val_classes==cls)]))[0]
                    indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                    indx_of_interest = indx_of_interest[:,None].expand(-1, val_features[cls]['features'].shape[1])
                    current_batch[cls] = val_features[cls]['features'].gather(0, indx_of_interest)
                logger.info(f"Running on validation data")

                event.clear()
                no_of_classes_to_process = len(set(val_classes[val_batch_nos==batch].tolist()))
                if args.no_multiprocessing or no_of_classes_to_process==1:
                    args.world_size = 1
                    common_operations.call_specific_approach(0, args, current_batch, completed_q, event, rolling_models)
                    p = None
                    results_for_all_batches[batch] = common_operations.convert_q_to_dict(args, completed_q, p, event)
                    args.world_size = torch.cuda.device_count()
                else:
                    args.world_size = min(no_of_classes_to_process, torch.cuda.device_count())
                    p = mp.spawn(common_operations.call_specific_approach,
                                 args=(args, current_batch, completed_q, event, rolling_models),
                                 nprocs=args.world_size, join=False)
                    results_for_all_batches[batch] = common_operations.convert_q_to_dict(args, completed_q, p, event)
                results_for_all_batches[batch]['classes_order'] = sorted(rolling_models.keys())


            args.output_dir=pathlib.Path(args.output_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logger.critical(f"Results for {new_classes_per_batch} new classes/batch DM {args.distance_multiplier}"
                            f" CT {args.cover_threshold}")

            acc_to_plot=[]
            batch_nos_to_plot = []
            for batch_no in sorted(results_for_all_batches.keys()):
                scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
                correct = 0.
                total = 0.
                for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
                    scores = results_for_all_batches[batch_no][test_cls]
                    total+=scores.shape[0]
                    max_indx = torch.argmax(scores,dim=1)
                    correct+=sum(scores_order[max_indx]==test_cls)
                acc = (correct/total)*100.
                acc_to_plot.append(acc)
                batch_nos_to_plot.append(scores_order.shape[0])
                logger.critical(f"Accuracy on Batch {batch_no} : {acc:.2f}")

            logger.critical(f"Average Accuracy {np.mean(acc_to_plot):.2f}")

            all_grid_search_results.append((f"{new_classes_per_batch:03d}", f"{np.mean(acc_to_plot):06.2f}",
                                            f"{args.distance_multiplier[0]}", f"{args.cover_threshold[0]}"))

        all_grid_search_results = sorted(all_grid_search_results)
        per_class_best_results.append(' '.join(all_grid_search_results[-1]))
        tabular_results.append(all_grid_search_results[-1][1])
        logger.critical("Grid Search Results:\n"+'\n'.join((' '.join(_) for _ in all_grid_search_results)))
    logger.critical("Best Results:\n"+'\n'.join(per_class_best_results))

    if len(all_new_classes_per_batch)>1:
        logger.critical(' & '.join(tabular_results))