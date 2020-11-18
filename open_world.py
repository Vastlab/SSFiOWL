"""
1) Create protocol
2) Read all features
If rolling_models keys is zero, learn
else prict unknowness

With the first batch or the initialization batch initialize the known classes
With the

For each training batch detect unknowns based on previous models
Use the unknowns to learn a new class and EVM models
Get accuracy on validation set

5) Rearrange all read features according to batch orders
3) Arrange features into batches dictonary format
4) For the initialization batch run the approach


Read all features for ImageNet Images from MoCoV2 network
2) Concatenate all Images Features, their respective Image Names and their class names
3) Cluster all features using one of the clustering algorithms
4) Get the order of all the images and samples combinations in batches
5) Rearrange all read features according to batch orders
6) For each batch get start and end indexes in all features
7) Iterate over each batch to perform the OOD, accumulation and incremental steps
"""

import argparse
import torch
import time
import protocols
import data_prep
import numpy as np
import utils
import common_operations
import viz
import torch.multiprocessing as mp
import pickle
import pathlib
import accumulation_algos
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# torch.set_num_threads(32)
def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script trains an EVM",
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


    parser.add_argument('--accumulation_algo', default='mimic_incremental', type=str,
                        help='Name of the accumulation algorithm to use',
                        choices=['mimic_incremental','learn_new_unknowns','update_existing_learn_new'])
    parser.add_argument("--unknowness_threshold", help="unknowness probability score above which a sample is considered as unknown",
                        type=float, default=0.5)

    parser.add_argument('--OOD_Algo', default='OpenMax', type=str,
                        help='Name of the Out of Distribution detection algorithm',
                        choices=['OpenMax','EVM','MultiModalOpenMax'])

    # weibull_params = parser.add_subparsers(help='Parameters for weibull specific methods')
    # weibull_params_parser = weibull_params.add_parser('weibull_params', help='a help')
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

    parser.add_argument('--Clustering_Algo', default='KMeans', type=str,
                        help='Name of the clustering algorithm for multimodal openmax',
                        choices=['dbscan','KMeans','finch'])
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

# @utils.time_recorder
# def main():
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

    # TODO
    # val_batch_nos, val_images, val_classes = protocols.ImageNetIncremental(files_to_add = ['imagenet_1000_val'],
    #                                                                        initial_no_of_classes=args.initialization_classes,
    #                                                                        new_classes_per_batch=args.new_classes_per_batch,
    #                                                                        total_classes=args.total_no_of_classes)

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


            # Find Unknown Detection Accuracy

            # C+1 Class Classification Accuracy


        # Accumulate all unknown samples
        accumulated_samples = get_learning_samples(args, current_batch, rolling_models, probabilities_for_train_set)

        # Add exemplars
        no_of_exemplar_batches = 0
        if batch!=0 and args.no_of_exemplars!=0 and not args.all_samples:
            print(f"Finding and Adding exemplars to negatives")
            current_batch_size=0
            accumulated_samples[f'exemplars_{no_of_exemplar_batches}'] = []
            for cls_name in rolling_models:
                torch.manual_seed(0)
                random.seed(0)
                np.random.seed(0)
                ind_of_interest = torch.randint(rolling_models[cls_name]['extreme_vectors'].shape[0],
                                                (min(args.no_of_exemplars,
                                                     rolling_models[cls_name]['extreme_vectors'].shape[0]),
                                                 1))
                current_exemplars = features[cls_name]['features'].gather(0,ind_of_interest.expand(
                                                            -1,rolling_models[cls_name]['extreme_vectors'].shape[1]))
                accumulated_samples[f'exemplars_{no_of_exemplar_batches}'].append(current_exemplars)
                current_batch_size+=current_exemplars.shape[0]
                if current_batch_size>=1000:
                    accumulated_samples[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(accumulated_samples[f'exemplars_{no_of_exemplar_batches}'])
                    no_of_exemplar_batches+=1
                    accumulated_samples[f'exemplars_{no_of_exemplar_batches}'] = []
                    current_batch_size=0
            if type(accumulated_samples[f'exemplars_{no_of_exemplar_batches}']) == list:
                if len(accumulated_samples[f'exemplars_{no_of_exemplar_batches}'])==0:
                    del accumulated_samples[f'exemplars_{no_of_exemplar_batches}']
                else:
                    accumulated_samples[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(accumulated_samples[f'exemplars_{no_of_exemplar_batches}'])
                    no_of_exemplar_batches+=1


        # Run enrollment for unknown samples probabilities_for_train_set
        print(f"######################### Enrolling {len(accumulated_samples)-no_of_exemplar_batches} new classes #########################")
        event.clear()
        if args.no_multiprocessing:
            args.world_size = 1
            common_operations.call_specific_approach(0, args, accumulated_samples, completed_q, event)
            p=None
            models = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        else:
            p = mp.spawn(common_operations.call_specific_approach,
                         args=(args, accumulated_samples, completed_q, event),
                         nprocs=args.world_size,
                         join=False)
            models = utils.convert_q_to_dict(args, completed_q, p, event)
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

        print(f"$$$$$$$$$$$$$$$$$$$$ len of rolling_models {len(rolling_models)} $$$$$$$$$$$$$$$$$$$$")

    dir_name = f"{args.total_no_of_classes}_{args.initialization_classes}_{args.new_classes_per_batch}"
    file_name = f"{args.distance_metric}_{args.Clustering_Algo}_{args.tailsize}_{args.cover_threshold}_{args.distance_multiplier}"
    if args.all_samples:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/all_samples/")
    else:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/no_of_exemplars_{args.no_of_exemplars}/")
    file_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to path {file_path}")
    pickle.dump(results_for_all_batches, open(f"{file_path}/{args.OOD_Algo}_{file_name}.p", "wb"))

    try:
        CCA = []
        for batch_no in sorted(results_for_all_batches.keys()):
            scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
            correct = 0.
            total = 0.
            for test_cls in scores_order.tolist():
                scores = results_for_all_batches[batch_no][test_cls]
                total+=scores.shape[0]
                max_indx = torch.argmax(scores,dim=1)
                correct+=sum(scores_order[max_indx]==test_cls)
            CCA.append((correct/total)*100.)
            print(f"Accuracy on Batch {batch_no} : {CCA[-1]}")
        print(f"Average Closed Set Classification Accuracy : {np.mean(CCA)}")

        UDA = []
        OCA = []
        for batch_no in sorted(results_for_all_batches.keys()):
            unknown_classes = (set(results_for_all_batches[batch_no].keys()) -
                               set(results_for_all_batches[batch_no]['classes_order']) -
                               set(['classes_order']))
            scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
            UDA_correct = 0.
            OCA_correct = 0.
            total = 0.
            UDA_total = 0.
            for test_cls in list(set(results_for_all_batches[batch_no].keys()) - set(['classes_order'])):
                scores = results_for_all_batches[batch_no][test_cls]
                total+=scores.shape[0]
                max_scores = torch.max(scores,dim=1)
                unknowness_scores= 1 - max_scores.values
                predicted_as_unknowns = sum(unknowness_scores>args.unknowness_threshold)
                if test_cls in unknown_classes:
                    UDA_correct+=predicted_as_unknowns
                    OCA_correct+=predicted_as_unknowns
                    UDA_total += scores.shape[0]
                else:
                    temp = np.array([scores_order[max_scores.indices[unknowness_scores <= args.unknowness_threshold]]])
                    if len(temp.shape)>1:
                        temp = np.squeeze(temp)
                    OCA_correct+=sum(temp == test_cls)
            UDA.append((UDA_correct/max(UDA_total,1))*100.)
            OCA.append((OCA_correct/total)*100.)
            print(f"Unknowness detection accuracy on Batch {batch_no} : {UDA[-1]}  OCA {OCA[-1]}")
        print(f"Average Unknowness Accuracy : {np.mean(UDA)} OCA {np.mean(OCA)}")
        print(f"For Tabeling")
        print(f"{np.mean(UDA)} & {np.mean(OCA)} & {np.mean(CCA)}")
        print(f"{round(np.mean(UDA).astype(np.float64),2)} & {round(np.mean(OCA),2)} & {round(np.mean(CCA),2)}")


        acc_to_plot=[]
        batch_nos_to_plot = []

        for batch_no in sorted(results_for_all_batches.keys()):
            scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
            correct = 0.
            total = 0.
            for test_cls in list(set(results_for_all_batches[batch_no].keys()) - set(['classes_order'])):
                scores = results_for_all_batches[batch_no][test_cls]
                total+=scores.shape[0]
                max_indx = torch.argmax(scores,dim=1)
                correct+=sum(scores_order[max_indx]==test_cls)
            acc = (correct/total)*100.
            acc_to_plot.append(acc)
            batch_nos_to_plot.append(scores_order.shape[0])
            print(f"Accuracy on Batch {batch_no} : {acc}")

        print(f"Average accuracy : {np.mean(acc_to_plot)}")
        # viz.plot_accuracy_vs_batch(acc_to_plot, batch_nos_to_plot,labels=args.OOD_Algo)
    except:
        from IPython import embed;embed();