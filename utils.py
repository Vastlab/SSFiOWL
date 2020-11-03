import functools
import time

def time_recorder(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        to_return = func(*args, **kwargs)
        print(f"Time taken to complete the function {func.__name__} : "
              f"{time.strftime('%H Hours %M Minutes %S seconds',time.gmtime(time.time() - start))}")
        return to_return
    return wrapper


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
            all_data.update((result,))
    event.set()
    if p is not None:
        p.join()
    return all_data
