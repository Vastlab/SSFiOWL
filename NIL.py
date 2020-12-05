import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
import pickle
import pathlib
import random
import protocols
import exemplar_selection
import data_prep
import utils
import common_operations
import viz

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""This script runs experiments for incremental learning 
                                                    i.e. Table 1 and 2 from the Paper"""
                                     )
    parser.add_argument("--training_feature_files",
                        nargs="+",
                        default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_train.hdf5"],
                        help="HDF5 feature files")
    parser.add_argument("--validation_feature_files",
                        nargs="+",
                        default=["/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"],
                        help="HDF5 feature files")
    parser.add_argument("--layer_names",
                        nargs="+",
                        help="Layer names to train EVM on",
                        default=["features"])
    parser.add_argument("--debug", help="debugging flag", action="store_true", default=False)
    parser.add_argument("--no_multiprocessing", help="debugging flag", action="store_true", default=False)

    parser.add_argument('--OOD_Algo', default='OpenMax', type=str,
                        help='Name of the Out of Distribution detection algorithm',
                        choices=['OpenMax','EVM','MultiModalOpenMax'])
    parser.add_argument('--Clustering_Algo', default='finch', type=str,
                        help='Clustering algorithm used for multi modal openmax',
                        choices=['KMeans','dbscan','finch'])

    parser.add_argument("--total_no_of_classes", help="total_no_of_classes", type=int, default=100)
    parser.add_argument("--initialization_classes", help="initialization_classes", type=int, default=50)
    parser.add_argument("--new_classes_per_batch", help="new_classes_per_batch", type=int, default=[1,2,5,10], nargs="+")

    parser.add_argument("--no_of_exemplars", help="no_of_exemplars",
                        type=int, default=0)
    parser.add_argument("--all_samples", help="all_samples", action="store_true", default=False)

    parser.add_argument("--tailsize", help="tail size to use",
                        type=float, default=33998.)
    parser.add_argument("--cover_threshold", help="cover threshold to use",
                        type=float, default=0.7)
    parser.add_argument("--distance_multiplier", help="distance multiplier to use",
                        type=float, default=0.55)
    parser.add_argument('--distance_metric', default='cosine', type=str,
                        help='distance metric to use', choices=['cosine','euclidean'])

    parser.add_argument('--port_no', default='9451', type=str,
                        help='port number for multiprocessing')
    parser.add_argument("--output_dir", help="output_dir", type=str, default='/scratch/adhamija/results/')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True

    all_new_classes_per_batch = args.new_classes_per_batch
    results=[]
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

        # Convert Each Batch into a dictionary where keys are class names
        event = mp.Event()
        rolling_models = {}
        results_for_all_batches = {}
        completed_q = mp.Queue()
        for batch in set(batch_nos.tolist()):
            print(f"Preparing training batch {batch}")
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
            print(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

            event.clear()
            no_of_classes_to_process = len(set(classes[batch_nos==batch].tolist())-set(rolling_models.keys()))
            if args.no_multiprocessing or no_of_classes_to_process==1:
                args.world_size = 1
                common_operations.call_specific_approach(0, args, current_batch, completed_q, event)
                p=None
                models = utils.convert_q_to_dict(args, completed_q, p, event)
            else:
                args.world_size = min(no_of_classes_to_process, torch.cuda.device_count())
                p = mp.spawn(common_operations.call_specific_approach,
                             args=(args, current_batch, completed_q, event),
                             nprocs=args.world_size,
                             join=False)
                models = utils.convert_q_to_dict(args, completed_q, p, event)

            print(f"Preparing validation data")
            rolling_models.update(models)
            current_batch = {}
            for cls in sorted(set(val_classes[val_batch_nos==batch].tolist())):
                indx_of_interest = np.where(np.in1d(val_features[cls]['images'], val_images[(val_batch_nos == batch) & (val_classes==cls)]))[0]
                indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
                indx_of_interest = indx_of_interest[:,None].expand(-1, val_features[cls]['features'].shape[1])
                current_batch[cls] = val_features[cls]['features'].gather(0, indx_of_interest)
            print(f"Running on validation data")

            event.clear()
            no_of_classes_to_process = len(set(val_classes[val_batch_nos==batch].tolist()))
            if args.no_multiprocessing or no_of_classes_to_process==1:
                args.world_size = 1
                common_operations.call_specific_approach(0, args, current_batch, completed_q, event, rolling_models)
                p = None
                results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
                args.world_size = torch.cuda.device_count()
            else:
                args.world_size = min(no_of_classes_to_process, torch.cuda.device_count())
                p = mp.spawn(common_operations.call_specific_approach,
                             args=(args, current_batch, completed_q, event, rolling_models),
                             nprocs=args.world_size, join=False)
                results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
            results_for_all_batches[batch]['classes_order'] = sorted(rolling_models.keys())


        dir_name = f"Incremental_Learning/InitialClasses-{args.initialization_classes}_TotalClasses-{args.total_no_of_classes}" \
                   f"_NewClassesPerBatch-{args.new_classes_per_batch}"
        file_name = f"{args.distance_metric}_EVMParams-{args.tailsize}_{args.cover_threshold}_{args.distance_multiplier}"
        if args.all_samples:
            file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/all_samples/")
        else:
            file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/no_of_exemplars_{args.no_of_exemplars}/")
        file_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to path {file_path}")
        pickle.dump(results_for_all_batches, open(f"{file_path}/{args.OOD_Algo}_{file_name}.pkl", "wb"))

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
            print(f"Accuracy on Batch {batch_no} : {acc:.2f}")

        print(f"Average Accuracy {np.mean(acc_to_plot):.2f}")
        results.append(f"{np.mean(acc_to_plot):.2f}")
    if len(all_new_classes_per_batch)>1:
        print(' & '.join(results))