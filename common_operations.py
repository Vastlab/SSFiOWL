import os
import utile
import torch

def convert_q_to_dict(args, completed_q, p=None, event=None):
    all_data = {}
    nb_ended_workers = 0
    k=0
    while nb_ended_workers < args.world_size:
        result = completed_q.get()
        k+=1
        if result[0] == "DONE":
            nb_ended_workers += 1
            print(f"Completed process {result[1]}/{args.world_size}")
        else:
            if result[0] not in all_data:
                all_data[result[0]]={}
            all_data[result[0]].update((result[1],))
    event.set()
    if p is not None:
        p.join()
    if len(all_data)==1:
        return all_data[list(all_data.keys())[0]]
    return all_data


def call_specific_approach(gpu, args, features_all_classes, completed_q, event, models=None):
    if args.world_size>1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port_no
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='env://',
            world_size=args.world_size,
            rank=gpu
        )
        torch.cuda.set_device(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    print(f"Started process {gpu}/{args.world_size}")
    class_names = list(features_all_classes.keys())
    exemplar_classes = []
    for _ in class_names:
        if 'exemplars_' in _:
            exemplar_classes.append(_)
    if len(exemplar_classes):
        print(" Removing Exemplars from positive classes to be processed ".center(90, '#'))
    class_names = sorted(list(set(class_names)-set(exemplar_classes)))
    div, mod = divmod(len(class_names), args.world_size)
    pos_classes_to_process = class_names[gpu * div + min(gpu, mod):(gpu + 1) * div + min(gpu + 1, mod)]
    print(f"Process {gpu} processing classes {pos_classes_to_process}")
    if models is None:
        OOD_Method = getattr(utile.opensetAlgos, args.OOD_Algo + '_Training')
    else:
        OOD_Method = getattr(utile.opensetAlgos, args.OOD_Algo + '_Inference')
    algorithm_results_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
    for current_class_output in algorithm_results_iterator:
        completed_q.put(current_class_output)
    completed_q.put(("DONE",gpu))
    if not args.no_multiprocessing and args.world_size>1:
        print(f"Process {gpu} is waiting now")
        event.wait()
    print(f"Shutting down process {gpu}")
    return