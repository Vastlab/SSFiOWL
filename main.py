"""
1) Create protocol
2) Read all features
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
torch.manual_seed(0)
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

    parser.add_argument('--OOD_Algo', default='OpenMax', type=str,
                        help='Name of the Out of Distribution detection algorithm',
                        choices=['OpenMax','EVM','MultiModalOpenMax'])

    """
    protocol_group = parser.add_subparsers(help='Parameters for the protocol')
    protocol_group_parser = protocol_group.add_parser('protocol_params', help='a help')
    protocol_group_parser.add_argument('--protocol_name', default='basic_protocol', type=str,
                                       help='protocol function name', choices=['basic_protocol'])
    protocol_group_parser.add_argument("--initial_no_of_classes", help="number of classes to initialize with",
                                       type=int, default=950)
    protocol_group_parser.add_argument("--new_classes_per_batch", help="number of classes to add per batch",
                                       type=int, default=10)
    protocol_group_parser.add_argument("--initial_batch_size", help="number of images in initialization batch",
                                       type=int, default=20000)
    protocol_group_parser.add_argument("--batch_size", help="number of images in every incremental batch",
                                       type=int, default=50)
    """

    # weibull_params = parser.add_subparsers(help='Parameters for weibull specific methods')
    # weibull_params_parser = weibull_params.add_parser('weibull_params', help='a help')
    parser.add_argument("--total_no_of_classes", help="total_no_of_classes", type=int, default=100)
    parser.add_argument("--initialization_classes", help="initialization_classes", type=int, default=50)
    parser.add_argument("--new_classes_per_batch", help="new_classes_per_batch", type=int, default=5)

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

# @utils.time_recorder
# def main():
if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    args = command_line_options()
    args.world_size = torch.cuda.device_count()
    if args.world_size==1:
        args.no_multiprocessing = True

    # Get the protocols
    batch_nos, images, classes = protocols.ImageNetIncremental(initial_no_of_classes=args.initialization_classes,
                                                               new_classes_per_batch=args.new_classes_per_batch,
                                                               total_classes=args.total_no_of_classes)
    val_batch_nos, val_images, val_classes = protocols.ImageNetIncremental(files_to_add = ['imagenet_1000_val'],
                                                                           initial_no_of_classes=args.initialization_classes,
                                                                           new_classes_per_batch=args.new_classes_per_batch,
                                                                           total_classes=args.total_no_of_classes)

    # Read all Features
    args.feature_files = args.training_feature_files
    features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(classes.tolist())))
    args.feature_files = args.validation_feature_files
    val_features = data_prep.prep_all_features_parallel(args, all_class_names=list(set(val_classes.tolist())))

    for _ in features:
        if features[_]['features'].shape[0]<=500:
            print(_, features[_]['features'].shape, features[_]['images'].shape)
    for _ in val_features:
        if val_features[_]['features'].shape[0]!=50:
            print(_, val_features[_]['features'].shape, val_features[_]['images'].shape)


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
            no_of_exemplar_batches=0
            current_batch_size=0
            current_batch[f'exemplars_{no_of_exemplar_batches}'] = []
            for cls_name in rolling_models:
                ind_of_interest = torch.randint(rolling_models[cls_name]['extreme_vectors'].shape[0],(min(args.no_of_exemplars,rolling_models[cls_name]['extreme_vectors'].shape[0]),1))
                current_exemplars = features[cls]['features'].gather(0,ind_of_interest.expand(
                                                            -1,rolling_models[cls_name]['extreme_vectors'].shape[1]))
                current_batch[f'exemplars_{no_of_exemplar_batches}'].append(current_exemplars)
                current_batch_size+=current_exemplars.shape[0]
                if current_batch_size>=1000:
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(current_batch[f'exemplars_{no_of_exemplar_batches}'])
                    no_of_exemplar_batches+=1
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = []
            if type(current_batch[f'exemplars_{no_of_exemplar_batches}']) == list:
                if len(current_batch[f'exemplars_{no_of_exemplar_batches}'])==0:
                    del current_batch[f'exemplars_{no_of_exemplar_batches}']
                else:
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(current_batch[f'exemplars_{no_of_exemplar_batches}'])
        # Add all negative samples
        if args.all_samples and batch!=0:
            no_of_exemplar_batches=0
            current_batch_size=0
            current_batch[f'exemplars_{no_of_exemplar_batches}'] = []
            for cls in sorted(set(rolling_models.keys())):
                current_exemplars = features[cls]['features']
                current_batch[f'exemplars_{no_of_exemplar_batches}'].append(current_exemplars)
                current_batch_size+=current_exemplars.shape[0]
                if current_batch_size>=1000:
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(current_batch[f'exemplars_{no_of_exemplar_batches}'])
                    no_of_exemplar_batches+=1
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = []
            if type(current_batch[f'exemplars_{no_of_exemplar_batches}']) == list:
                if len(current_batch[f'exemplars_{no_of_exemplar_batches}'])==0:
                    del current_batch[f'exemplars_{no_of_exemplar_batches}']
                else:
                    current_batch[f'exemplars_{no_of_exemplar_batches}'] = torch.cat(current_batch[f'exemplars_{no_of_exemplar_batches}'])

        for cls in sorted(set(classes[batch_nos==batch].tolist())-set(rolling_models.keys())):
            indx_of_interest = np.where(np.in1d(features[cls]['images'], images[(batch_nos == batch) & (classes==cls)]))[0]
            indx_of_interest = torch.tensor(indx_of_interest, dtype=torch.long)
            indx_of_interest = indx_of_interest[:,None].expand(-1, features[cls]['features'].shape[1])
            current_batch[cls] = features[cls]['features'].gather(0, indx_of_interest)
        print(f"Processing batch {batch}/{len(set(batch_nos.tolist()))}")

        # if batch!=0 and args.no_of_exemplars!=0:
        #     from IPython import embed;embed();

        event.clear()
        if args.no_multiprocessing:
            args.world_size = 1
            common_operations.each_process_trainer(0, args, current_batch, completed_q, event)
            p=None
            models = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        else:
        # if True:
            p = mp.spawn(common_operations.each_process_trainer,
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
        if args.no_multiprocessing:
            args.world_size = 1
            common_operations.each_process_trainer(0, args, current_batch, completed_q, event, rolling_models)
            p = None
            results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
            args.world_size = torch.cuda.device_count()
        else:
            p = mp.spawn(common_operations.each_process_trainer,
                         args=(args, current_batch, completed_q, event, rolling_models),
                         nprocs=args.world_size, join=False)
            results_for_all_batches[batch] = utils.convert_q_to_dict(args, completed_q, p, event)
        results_for_all_batches[batch]['classes_order'] = sorted(rolling_models.keys())

        # from IPython import embed;embed();

    dir_name = f"{args.total_no_of_classes}_{args.initialization_classes}_{args.new_classes_per_batch}"
    file_name = f"{args.distance_metric}_{args.Clustering_Algo}_{args.tailsize}_{args.cover_threshold}_{args.distance_multiplier}"
    if args.all_samples:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/all_samples/")
    else:
        file_path = pathlib.Path(f"{args.output_dir}/{dir_name}/no_of_exemplars_{args.no_of_exemplars}/")
    file_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to path {file_path}")
    pickle.dump(results_for_all_batches, open(f"{file_path}/{args.OOD_Algo}_{file_name}.p", "wb"))


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

    # viz.plot_accuracy_vs_batch(acc_to_plot, batch_nos_to_plot,labels=args.OOD_Algo)