import torch
import torch.distributed as dist
import os
import utils
import OOD



def call_specific_approach(gpu, args, features_all_classes, completed_q, event, models=None):
    """
    :param gpu:
    :param args:
    :param pos_classes_to_process: List of classes to be processed by the current process
    :param all_class_features:  Should be a dictionary with keys as class names and values as tensor
    :return:
    """
    if args.world_size>1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9451'
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            world_size=args.world_size,
            rank=gpu
        )
        torch.cuda.set_device(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    print(f"started process {gpu}/{args.world_size}")
    class_names = list(features_all_classes.keys())
    exemplar_classes = []
    for _ in class_names:
        if 'exemplars_' in _:
            exemplar_classes.append(_)
    class_names = sorted(list(set(class_names)-set(exemplar_classes)))
    div, mod = divmod(len(class_names), args.world_size)
    pos_classes_to_process = class_names[gpu * div + min(gpu, mod):(gpu + 1) * div + min(gpu + 1, mod)]
    print(f"pos_classes_to_process {pos_classes_to_process}")
    if models is None:
        OOD_Method = getattr(OOD, args.OOD_Algo)
    else:
        OOD_Method = getattr(OOD, args.OOD_Algo + '_Inference')
    for current_cls_name in pos_classes_to_process:
        current_class_output = OOD_Method(current_cls_name, features_all_classes, args, gpu, models)
        # print(f"[current_cls_name, current_class_output] {[current_cls_name, current_class_output]}")
        completed_q.put([current_cls_name, current_class_output])
    completed_q.put(("DONE",gpu))
    if not args.no_multiprocessing and args.world_size>1:
        print(f"{args.no_multiprocessing} args.world_size {args.world_size}")
        print(f"{gpu} is WAITING NOW")
        event.wait()
    print(f"ENDING {gpu}")
    return
