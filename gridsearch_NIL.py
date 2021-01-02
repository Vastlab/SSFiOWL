import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.multiprocessing as mp
import pickle
import pathlib
import random
import itertools
import protocols
import data_prep
import common_operations
import viz
from utile import opensetAlgos
import time
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     add_help=False, usage=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true", default=False, help="debugging flag\ndefault: %(default)s")
    parser.add_argument("--no_multiprocessing", action="store_true", default=False,
                        help="Use for debugging or running on single GPU\ndefault: %(default)s")
    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing\ndefault: %(default)s')
    parser.add_argument("--output_dir", type=str, default='/scratch/adhamija/results/', help="Results directory")
    parser.add_argument('--OOD_Algo', default='EVM', type=str, choices=['OpenMax','EVM','MultiModalOpenMax'],
                        help='Name of the openset detection algorithm\ndefault: %(default)s')
    parser = data_prep.params(parser)
    protocol_params = parser.add_argument_group('Protocol params')
    protocol_params.add_argument("--total_no_of_classes", type=int, default=50,
                                 help="Total no of classes\ndefault: %(default)s")
    protocol_params.add_argument("--initialization_classes", type=int, default=30,
                                 help="No of classes in first batch\ndefault: %(default)s")
    protocol_params.add_argument("--new_classes_per_batch", type=int, default=[2,5,10], nargs="+",
                                 help="No of new classes added per batch\nSince exemplars are not used for grid search"
                                      "it is not possible to use 1 class increment steps for EVM\ndefault: %(default)s")
    known_args, unknown_args = parser.parse_known_args()
    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [parser],formatter_class = argparse.RawTextHelpFormatter,
                                            usage=argparse.SUPPRESS,
                                            description = "This script runs experiments for incremental learning " 
                                                          "i.e. Table 1 and 2 from the Paper")
    parser, OOD_params_info = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)
    OOD_params_info['group_parser'].set_defaults(distance_multiplier=np.around(np.arange(0.1,1.2,0.1),2).tolist())
    OOD_params_info['group_parser'].set_defaults(cover_threshold=np.around(np.arange(0.1,0.9,0.1),2).tolist())
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True

    results_for_all_class_increments = []
    for exp_no, new_classes_per_batch in enumerate(args.new_classes_per_batch):
        # Get the protocols
        batch_nos, images, classes = protocols.ImageNetIncremental(initial_no_of_classes=args.initialization_classes,
                                                                   new_classes_per_batch=new_classes_per_batch,
                                                                   total_classes=args.total_no_of_classes)
        combined_for_shuffle = list(zip(batch_nos, images, classes))
        random.shuffle(combined_for_shuffle)
        batch_nos, images, classes = zip(*combined_for_shuffle)
        num_to_split_at = int(len(batch_nos)*.8)
        val_batch_nos = np.array(batch_nos[num_to_split_at:])
        val_images = np.asarray(images[num_to_split_at:], dtype='<U30')
        val_classes = np.asarray(classes[num_to_split_at:], dtype='<U30')
        batch_nos = np.array(batch_nos[:num_to_split_at])
        images = np.asarray(images[:num_to_split_at], dtype='<U30')
        classes = np.asarray(classes[:num_to_split_at], dtype='<U30')

        # Read all Features
        if exp_no==0:
            args.feature_files = args.training_feature_files
            features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(classes.tolist())))

        rolling_models = {}
        classes_already_enrolled = []
        for batch in set(batch_nos.tolist()):

            print(f"Preparing training batch {batch}")
            training_batch = {}
            for cls in set(classes[batch_nos == batch].tolist())-set(classes_already_enrolled):
                indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes == cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:, None].expand(-1, features[cls]['features'].shape[1])
                training_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
            print(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

            # Convert Each Batch into a dictionary where keys are class names
            event = mp.Event()
            completed_q = mp.Queue()
            event.clear()
            no_of_classes_to_process = len(set(classes[batch_nos==batch].tolist())-set(classes_already_enrolled))
            classes_already_enrolled=set(classes[batch_nos==batch].tolist())
            if args.no_multiprocessing or no_of_classes_to_process==1:
                args.world_size = 1
                common_operations.call_specific_approach(0, args, training_batch, completed_q, event)
                p=None
                models = common_operations.convert_q_to_dict(args, completed_q, p, event)
            else:
                args.world_size = min(no_of_classes_to_process, torch.cuda.device_count())
                p = mp.spawn(common_operations.call_specific_approach,
                             args=(args, training_batch, completed_q, event),
                             nprocs=args.world_size,
                             join=False)
                models = common_operations.convert_q_to_dict(args, completed_q, p, event)

            rolling_models[batch]=models

        # Run validation on each batch for each parameter combination
        results_to_read = []
        for param_combination in rolling_models[0].keys():
            models={}
            results_for_all_batches = {}
            acc_to_plot=[]
            batch_nos_to_plot = []
            for batch in set(batch_nos.tolist()):
                models.update(rolling_models[batch][param_combination])
                print(f"Preparing validation data for batch {batch}")
                validation_batch = {}
                for cls in sorted(set(val_classes[val_batch_nos == batch].tolist())):
                    indx_of_interest = \
                    np.where(np.in1d(features[cls]['images'], val_images[(val_batch_nos == batch) & (val_classes == cls)]))[0]
                    indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                    indx_of_interest = indx_of_interest[:, None].expand(-1, features[cls]['features'].shape[1])
                    validation_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
                print(f"Running validation for batch {batch}")

                event.clear()
                if args.no_multiprocessing or len(validation_batch.keys())==1:
                    args.world_size = 1
                    common_operations.call_specific_approach(0, args, validation_batch, completed_q, event, models)
                    p = None
                    results_for_all_batches[batch] = common_operations.convert_q_to_dict(args, completed_q, p, event)
                    args.world_size = torch.cuda.device_count()
                else:
                    args.world_size = min(len(validation_batch.keys()), torch.cuda.device_count())
                    p = mp.spawn(common_operations.call_specific_approach,
                                 args=(args, validation_batch, completed_q, event, models),
                                 nprocs=args.world_size, join=False)
                    results_for_all_batches[batch] = common_operations.convert_q_to_dict(args, completed_q, p, event)
                results_for_all_batches[batch]['classes_order'] = sorted(models.keys())

                # Compute accuracy for the validation set on the current batch
                scores_order = np.array(sorted(results_for_all_batches[batch]['classes_order']))
                correct = 0.
                total = 0.
                for test_cls in list(set(results_for_all_batches[batch].keys()) - {'classes_order'}):
                    scores = results_for_all_batches[batch][test_cls]
                    total+=scores.shape[0]
                    max_indx = torch.argmax(scores,dim=1)
                    correct+=sum(scores_order[max_indx]==test_cls)
                acc = (correct/total)*100.
                acc_to_plot.append(acc)
                batch_nos_to_plot.append(scores_order.shape[0])
                print(f"Accuracy on Batch {batch} : {acc:.2f}")

            print(f"Average Accuracy {np.mean(acc_to_plot):.2f}")
            results_to_read.append([f"{np.mean(acc_to_plot):.2f}",
                                    f"{param_combination}"])
            results_to_read = sorted(results_to_read)
            print(f"{results_to_read}")
        with open(f"{args.output_dir}/{time.strftime('%Y%m%d-%H%M%S')}_grid_search_results_{new_classes_per_batch}.txt",
                  'w') as f:
            for item in results_to_read:
                f.write("%s\n" % item)
        results_for_all_class_increments.append((new_classes_per_batch,results_to_read[-1]))
    print(f"Best combinations {results_for_all_class_increments}")