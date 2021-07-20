import argparse
import pathlib
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import protocols
import exemplar_selection
import data_prep
import common_operations
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
    parser.add_argument('--OOD_Algo', default='MLP', type=str, choices=['OpenMax','EVM','MultiModalOpenMax','MLP'],
                        help='Name of the approach')

    parser = data_prep.params(parser)

    protocol_params = parser.add_argument_group('Protocol params')
    protocol_params.add_argument("--use_cub200", default=False, action="store_true",
                                 help="Use Cub200 protocol instead of ImageNet\ndefault: %(default)s")
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
    if known_args.OOD_Algo!="MLP":
        parser, _ = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    else:
        MLP_params_parser = params_parser.add_argument_group(title="MLP", description="MLP params")
        MLP_params_parser.add_argument("--lr", nargs="+", type=float, default=[1e-2, 1e-3],
                                       help="Learning rate to use at various learning batches."
                                            "If two lr are provided they correspond to 1st and rest of the batches.")
        MLP_params_parser.add_argument("--epochs", nargs="+", type=int, default=[300],
                                       help="Number of epochs to train for each batch."
                                            "If two numbers are provided they correspond to the 1st and rest")
        parser = params_parser
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    args = command_line_options()
    
    # Does not support multi-gpus for now
    args.world_size = 1
    args.no_multiprocessing = True

    if args.debug:
        args.verbose = 0
    logger = vastlogger.setup_logger(level=args.verbose, output=args.output_dir)
    all_new_classes_per_batch = args.new_classes_per_batch

    event = mp.Event()

    for exp_no, new_classes_per_batch in enumerate(all_new_classes_per_batch):
        args.new_classes_per_batch = new_classes_per_batch

        # Get the protocols
        if args.use_cub200:
            batch_nos, images, classes = protocols.cub200()
            val_batch_nos, val_images, val_classes = protocols.cub200()
        else:
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

        results_for_all_batches = {}
        completed_q = mp.Queue()
        stored_exemplars = {}
        current_batch = {}
        models_across_batches = {}
        for batch in set(batch_nos.tolist()):
            logger.info(f"Preparing training batch {batch}")

            exemplars_to_add = None
            # Add exemplars
            if batch!=0 and not args.all_samples:
                exemplars_to_add = exemplar_selection.random_selector(current_batch, [*models_across_batches],
                                                                      no_of_exemplars=args.no_of_exemplars)
            # Add all negative samples
            if args.all_samples and batch!=0:
                exemplars_to_add = exemplar_selection.add_all_negatives(current_batch, [*models_across_batches])

            if exemplars_to_add is not None:
                for e in exemplars_to_add:
                    if e not in stored_exemplars:
                        stored_exemplars[e] = exemplars_to_add[e]
                        if exemplars_to_add[e].shape[0]==0:
                            if e in stored_exemplars: del stored_exemplars[e]
                            if e in current_batch: del current_batch[e]
                current_batch.update(stored_exemplars)

            new_classes_to_add = sorted(set(classes[batch_nos==batch].tolist())-set([*models_across_batches]))
            for cls in new_classes_to_add:
                indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes==cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:,None].expand(-1, features[cls]['features'].shape[1])
                current_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)

            logger.info(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

            no_of_classes_to_process = len(set(classes[batch_nos==batch].tolist()))

            common_operations.call_specific_approach(0, batch, args, current_batch, completed_q, event,
                                                     new_classes_to_add = new_classes_to_add)
            model = common_operations.convert_q_to_dict(args, completed_q, None, event)

            logger.info(f"Preparing validation data")
            current_validation_batch = {}
            for cls in sorted(set(val_classes[val_batch_nos==batch].tolist())):
                indx_of_interest = np.where(np.in1d(val_features[cls]['images'], val_images[(val_batch_nos == batch) & (val_classes==cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:,None].expand(-1, val_features[cls]['features'].shape[1])
                current_validation_batch[cls] = val_features[cls]['features'].gather(0, indx_of_interest)

            results_for_all_batches[batch] = {}
            if args.OOD_Algo == "MLP":
                models_across_batches = dict.fromkeys(model.cls_names)
                results_for_all_batches[batch]['classes_order'] = model.cls_names
            else:
                models_across_batches.update(model)
                results_for_all_batches[batch]['classes_order'] = sorted([*models_across_batches])
            logger.info(f"Running on validation data")
            common_operations.call_specific_approach(0, batch, args, current_validation_batch, completed_q, event,
                                                     models_across_batches)
            results_for_all_batches[batch].update(common_operations.convert_q_to_dict(args,completed_q,
                                                                                      None, event))

        args.output_dir=pathlib.Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.critical(f"Results for {new_classes_per_batch} class increments")

        UDA, OCA, CCA = eval.calculate_CCA_on_thresh(results_for_all_batches, threshold=0.)