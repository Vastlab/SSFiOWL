import utils
import math
import h5py
import torch
import torch.multiprocessing as mp
from functools import partial
import numpy as np

def read_features(args, feature_file_names=None, cls_to_process=None):
    try:
        h5_objs = [h5py.File(file_name, "r") for file_name in feature_file_names]
        file_layer_comb = list(zip(h5_objs, args.layer_names))
        if cls_to_process is None:
            cls_to_process = sorted(list(h5_objs[0].keys()))
        if args.debug:
            cls_to_process = cls_to_process[:50]
        for cls in cls_to_process:
            temp = []
            for hf, layer_name in file_layer_comb:
                temp.append(torch.squeeze(torch.tensor(hf[cls][layer_name])))
            image_names = hf[cls]['image_names'][()].tolist()
            features = torch.cat(temp,dim=1)
            yield cls, features, image_names
    finally:
        for h in h5_objs:
            h.close()

def prep_single_chunk(args, cls_to_process):
    features_gen = read_features(args,
                                 feature_file_names = args.feature_files,
                                 cls_to_process = cls_to_process)
    data_to_return = {}
    for cls, feature, image_names in features_gen:
        data_to_return[cls] = {}
        data_to_return[cls]['images'] = np.array([f"{cls}/{_.decode('ascii')}" for _ in image_names], dtype='<U30')
        data_to_return[cls]['features'] = feature
    return data_to_return

@utils.time_recorder
def prep_all_features_parallel(args, all_class_names=None):
    for f in args.feature_files:
        assert Path(f).is_file(),f"File {f} does not exist"
    if all_class_names is None:
        with h5py.File(args.feature_files[0], "r") as hf:
            all_class_names = sorted(list(hf.keys()))
    cls_per_chunk = max(len(all_class_names)//(mp.cpu_count()-30),5)
    if args.debug:
        all_class_names = all_class_names[:100]
        cls_per_chunk = 2
    all_class_batches = [all_class_names[i:i+cls_per_chunk] for i in range(0, len(all_class_names), cls_per_chunk)]
    p = mp.Pool(min(30, mp.cpu_count()))
    all_data = p.map(partial(prep_single_chunk, args), all_class_batches)
    all_classes={}
    for data_returned in all_data:
        all_classes.update(data_returned)
    return all_classes