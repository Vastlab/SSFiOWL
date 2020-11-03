import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import utils
import OOD
import time
# mp.set_start_method('spawn')

def each_process_trainer(gpu, args, features_all_classes, completed_q, event, models=None):
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
    # time.sleep(gpu*25)

    print(f"started process {gpu}/{args.world_size}")
    class_names = list(features_all_classes.keys())
    exemplar_classes = []
    for _ in class_names:
        if 'exemplars_' in _:
            exemplar_classes.append(_)
    class_names = sorted(list(set(class_names)-set(exemplar_classes)))
    if len(class_names)%args.world_size == 0:
        cls_per_process = len(class_names) // args.world_size
    else:
        cls_per_process = len(class_names) // args.world_size + 1
    pos_classes_to_process = class_names[gpu*cls_per_process:(gpu+1)*cls_per_process]
    print(f"pos_classes_to_process {pos_classes_to_process}")
    print(f"S {gpu*cls_per_process} E {(gpu+1)*cls_per_process}")
    if models is None:
        OOD_Method = getattr(OOD, args.OOD_Algo)
    else:
        OOD_Method = getattr(OOD, args.OOD_Algo + '_Inference')
    for current_cls_name in pos_classes_to_process:
        current_class_output = OOD_Method(current_cls_name, features_all_classes, args, gpu, models)
        # print(f"[current_cls_name, current_class_output] {[current_cls_name, current_class_output]}")
        completed_q.put([current_cls_name, current_class_output])
    completed_q.put(("DONE",gpu))
    print(f"{gpu} is WAITING NOW")
    if not args.no_multiprocessing:
        event.wait()
    print(f"ENDING {gpu}")
    return



def each_process_inferencing(args, features_all_classes, gpu, models):
    """
    :param gpu:
    :param args:
    :param pos_classes_to_process: List of classes to be processed by the current process
    :param all_class_features:  Should be a dictionary with keys as class names and values as tensor
    :return:
    """
    class_names = list(features_all_classes.keys())
    cls_per_process = len(class_names) // args.world_size + 1
    pos_classes_to_process = class_names[gpu*cls_per_process:(gpu+1)*cls_per_process]
    OOD_Method = getattr(OOD, args.OOD_Algo+'_Inference')
    completed_q = []
    c=0
    for pos_cls_name in pos_classes_to_process:
        current_class_predictions = OOD_Method(pos_cls_name, features_all_classes, args, models, gpu)
        completed_q.append((pos_cls_name, current_class_predictions))
        c+=1
    completed_q.put((None))
    return


@utils.time_recorder
def multiprocess_spawner(gpu):
    if args.world_size>1:
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            world_size=args.world_size,
            rank=gpu
        )
    torch.cuda.set_device(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    # mp.Event()

def multiprocess_spawner(features_all_classes, args, models=None):
    if models is None:
        func = each_process_trainer
    else:
        func = each_process_inferencing
    # Running for single gpu mode
    completed_q = func(args, features_all_classes, 0, models)
    # if args.world_size == 1:
    #     _ = each_process_trainer(args, features_all_classes, 0)
    # Running for multi gpu mode
    # else:
    #     p = mp.Pool(args.world_size)
    #     _ = p.map(functools.partial(each_process_trainer, args, features_all_classes),range(args.world_size))
        # processes = mp.spawn(each_process_trainer, args=(args, features_all_classes, completed_q), nprocs=args.world_size, join=False)

    all_data = {}
    all_data.update(completed_q)
    """
    nb_ended_workers = 0
    i = 0
    while nb_ended_workers != args.world_size:
        # models = completed_q.get()
        models = completed_q[i]
        i+=1
        if models is None:
            nb_ended_workers += 1
        else:
            all_models.update(models)

    if args.world_size > 1:
        for p in processes:
            p.join()
    """
    return all_data
