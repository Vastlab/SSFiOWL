import argparse
import pickle
import pathlib
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import protocols
import data_prep
import utils
import common_operations
import exemplar_selection
import eval
import accumulation_algos
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def command_line_options():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     description = """This script runs the open world experiments from the paper
                                                      i.e. Table 3""")
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

    parser.add_argument('--accumulation_algo', default='mimic_incremental', type=str,
                        help='Name of the accumulation algorithm to use',
                        choices=['mimic_incremental','learn_new_unknowns','update_existing_learn_new'])
    parser.add_argument("--unknowness_threshold", help="unknowness probability score above which a sample is considered as unknown",
                        type=float, default=0.5)
    parser.add_argument('--OOD_Algo', default='OpenMax', type=str,
                        help='Name of the Out of Distribution detection algorithm',
                        choices=['OpenMax','EVM','MultiModalOpenMax'])
    parser.add_argument('--Clustering_Algo', default='finch', type=str,
                        help='Clustering algorithm used for multi modal openmax',
                        choices=['KMeans','dbscan','finch'])

    parser.add_argument("--total_no_of_classes", help="total_no_of_classes", type=int, default=100)
    parser.add_argument("--initialization_classes", help="initialization_classes", type=int, default=50)
    parser.add_argument("--new_classes_per_batch", help="new_classes_per_batch", type=int, default=5)
    parser.add_argument("--known_sample_per_batch", help="known_sample_per_batch", type=int, default=2500)
    parser.add_argument("--unknown_sample_per_batch", help="unknown_sample_per_batch", type=int, default=2500)
    parser.add_argument("--initial_no_of_samples", help="initial_no_of_samples", type=int, default=15000)

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

def get_learning_samples(args, current_batch,rolling_models, probabilities_for_train_set):
    accumulation_algo = getattr(accumulation_algos, args.accumulation_algo)
    return accumulation_algo(args, current_batch,rolling_models, probabilities_for_train_set)

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True

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

    event = mp.Event()
    rolling_models = {}
    results_for_all_batches = {}
    completed_q = mp.Queue()
    list_of_all_batch_nos = set(batch_nos.tolist())
    for batch in list_of_all_batch_nos:
        print(f"Preparing batch {batch} from training data (initialization/operational)")
        current_batch = get_current_batch(classes, features, batch_nos, batch, images)

        print(f"Processing batch {batch}/{len(list_of_all_batch_nos)}")
        probabilities_for_train_set={}
        if len(rolling_models.keys())>0:
            print(f"Getting probabilities for the current operational batch")
            event.clear()
            if args.no_multiprocessing:
                args.world_size = 1
                common_operations.call_specific_approach(0, args, current_batch, completed_q, event, rolling_models)
                p = None
                probabilities_for_train_set = utils.convert_q_to_dict(args, completed_q, p, event)
                args.world_size = torch.cuda.device_count()
            else:
                p = mp.spawn(common_operations.call_specific_approach,
                             args=(args, current_batch, completed_q, event, rolling_models),
                             nprocs=args.world_size, join=False)
                probabilities_for_train_set = utils.convert_q_to_dict(args, completed_q, p, event)
            probabilities_for_train_set['classes_order'] = sorted(rolling_models.keys())

        # Accumulate all unknown samples
        accumulated_samples = get_learning_samples(args, current_batch, rolling_models, probabilities_for_train_set)

        # Add exemplars
        exemplars_to_add={}
        if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
            exemplars_to_add = exemplar_selection.random_selector(features, rolling_models,
                                                                  no_of_exemplars=args.no_of_exemplars)
            accumulated_samples.update(exemplars_to_add)

        # Run enrollment for unknown samples probabilities_for_train_set
        no_of_classes_to_enroll = len(accumulated_samples) - len(exemplars_to_add)
        print(f"{f' Enrolling {no_of_classes_to_enroll} new classes with {len(exemplars_to_add)} exemplar batches '.center(90, '#')}")
        event.clear()
        if args.no_multiprocessing or  no_of_classes_to_enroll== 1:
            args.world_size = 1
            common_operations.call_specific_approach(0, args, accumulated_samples, completed_q, event)
            p=None
            models = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        else:
            args.world_size = min(no_of_classes_to_enroll, torch.cuda.device_count())
            p = mp.spawn(common_operations.call_specific_approach,
                         args=(args, accumulated_samples, completed_q, event),
                         nprocs=args.world_size,
                         join=False)
            models = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        rolling_models.update(models)

        print(f"Preparing validation data")
        current_batch = get_current_batch(val_classes, val_features, val_batch_nos, 0, val_images,
                                          classes_to_fetch=set(classes[batch_nos<=min(batch+1,max(batch_nos))].tolist()))

        print(f"Running on validation data")
        event.clear()
        if args.no_multiprocessing:
            args.world_size = 1
            common_operations.call_specific_approach(0, args, current_batch, completed_q, event, rolling_models)
            p = None
            results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        else:
            p = mp.spawn(common_operations.call_specific_approach,
                         args=(args, current_batch, completed_q, event, rolling_models),
                         nprocs=args.world_size, join=False)
            results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
        results_for_all_batches[batch]['classes_order'] = sorted(rolling_models.keys())

    dir_name = f"OpenWorld_Learning/InitialClasses-{args.initialization_classes}_TotalClasses-{args.total_no_of_classes}" \
               f"_NewClassesPerBatch-{args.new_classes_per_batch}"
    file_name = f"{args.distance_metric}_EVMParams-{args.tailsize}_{args.cover_threshold}_{args.distance_multiplier}"
    if args.all_samples:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/all_samples/")
    else:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/no_of_exemplars_{args.no_of_exemplars}/")
    file_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to path {file_path}")
    pickle.dump(results_for_all_batches, open(f"{file_path}/{args.OOD_Algo}_{file_name}.pkl", "wb"))
    CCA = eval.calculate_CCA(results_for_all_batches)
    UDA, OCA, _ = eval.calculate_UDA_OCA(results_for_all_batches, unknowness_threshold=args.unknowness_threshold)
    print(f"For Tabeling")
    print(f"{np.mean(UDA):.2f} & {np.mean(OCA):.2f} & {np.mean(CCA):.2f}")