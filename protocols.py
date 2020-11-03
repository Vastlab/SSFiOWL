import csv
import random
import utils
import functools
import numpy as np

def protocol(func):
    @functools.wraps(func)
    def wrapper(*args, files_to_add = ['imagenet_1000_train'], **kwargs):
        # files_to_add=['imagenet_1000_train.csv','imagenet_1000_val.csv','imagenet_360.csv']
        all_images = []
        for file_name in files_to_add:
            with open(f"/home/jschwan2/simclr-converter/{file_name}.csv", "r") as f:
                images = list(csv.reader(f, delimiter=","))
            images = [(i.split('/')[1], '/'.join(i.split('/')[1:])) for i, _ in images]
            all_images.extend(images)
        sorted(all_images)
        all_batches = func(*args, **kwargs, all_images=all_images)
        batch_nos, images, classes = zip(*all_batches)
        batch_nos = np.array(batch_nos)
        images = np.array(images)
        classes = np.array(classes)
        return batch_nos, images, classes
    return wrapper

@utils.time_recorder
@protocol
def basic_protocol(initial_no_of_classes=200,
                   new_classes_per_batch=10,
                   initial_batch_size=20000,
                   batch_size=500,
                   all_images=[], **kwargs):
    all_batches=[]
    class_names, image_names = zip(*all_images)
    class_names = list(set(class_names))
    random.seed(9)
    random.shuffle(class_names)
    for batch_no, no_classes_in_current_batch in enumerate(range(initial_no_of_classes,
                                                                 len(class_names)+1,
                                                                 new_classes_per_batch)):
        classes_of_interest = class_names[:no_classes_in_current_batch]
        random.shuffle(all_images)
        images_of_interest = []
        images_to_remove = []
        for cls_name, image in all_images:
            if cls_name in classes_of_interest:
                images_of_interest.append((batch_no, image, cls_name))
                images_to_remove.append((cls_name, image))
            if (batch_no!=0 and len(images_to_remove)==batch_size):
                break
            if (batch_no==0 and len(images_to_remove)==initial_batch_size):
                break
        all_images = list(set(all_images) - set(images_to_remove))
        all_batches.extend(images_of_interest)
        print(f"Created batch no {batch_no} with {len(images_of_interest)} number of samples")
    return all_batches


@utils.time_recorder
@protocol
def ImageNetIncremental(initial_no_of_classes=50,
                        new_classes_per_batch=5, #10 #25
                        total_classes=100,
                        all_images=[]):
    all_batches=[]
    class_names, image_names = zip(*all_images)
    class_names = sorted(list(set(class_names)))
    # class_names = list(zip(range(len(class_names)), class_names))

    np.random.seed(1993)
    class_names = np.random.choice(class_names, total_classes, replace=False)
    
    # sno, class_names = zip(*class_names)
    # print(f"class_names {sno}")

    for batch_no, no_classes_in_current_batch in enumerate(range(initial_no_of_classes,
                                                                 len(class_names)+1,
                                                                 new_classes_per_batch)):
        classes_of_interest = class_names[:no_classes_in_current_batch]
        images_of_interest = []
        for cls_name, image in all_images:
            if cls_name in classes_of_interest:
                images_of_interest.append((batch_no, image, cls_name))
        all_batches.extend(images_of_interest)
        print(f"Created batch no {batch_no} with {len(images_of_interest)} number of samples")
    return all_batches


if __name__ == "__main__":
    initial_no_of_classes=(50,500)
    total_no_of_classes = (100, 1000)
    no_of_batches=(5,10,25)
    for initial, total in zip(initial_no_of_classes,total_no_of_classes):
        for batches in no_of_batches:
            no_of_new_classes_in_batch = (total-initial)//batches
            batch_nos, images, classes = protocols.ImageNetIncremental(initial_no_of_classes=initial,
                                                                       new_classes_per_batch=no_of_new_classes_in_batch,
                                                                       total_classes=total)
            val_batch_nos, val_images, val_classes = protocols.ImageNetIncremental(files_to_add = ['imagenet_1000_val'],
                                                                                   initial_no_of_classes=initial,
                                                                                   new_classes_per_batch=no_of_new_classes_in_batch,
                                                                                   total_classes=total)
