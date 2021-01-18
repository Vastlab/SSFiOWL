import csv
import random
from vast.tools import logger as vastlogger
import functools
import itertools
import numpy as np
logger = vastlogger.get_logger()

def protocol(func):
    @functools.wraps(func)
    def wrapper(*args, files_to_add = ['imagenet_1000_train'], **kwargs):
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
        images = np.array(images, dtype='<U30')
        classes = np.array(classes, dtype='<U30')
        return batch_nos, images, classes
    return wrapper

@vastlogger.time_recorder
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


@vastlogger.time_recorder
@protocol
def open_world_protocol(initial_no_of_classes=50,
                        new_classes_per_batch=10,
                        initial_batch_size=15000,
                        known_sample_per_batch=2500,
                        unknown_sample_per_batch=2500,
                        total_classes = 100,
                        all_images=[], **kwargs):
    all_batches=[]
    org_class_names, image_names = zip(*all_images)
    class_names = sorted(list(set(org_class_names)))
    org_class_names = np.array(org_class_names)
    image_names = np.array(image_names)

    np.random.seed(1993)
    class_names = np.random.choice(class_names, total_classes, replace=False)
    image_names = image_names[np.in1d(org_class_names, class_names)]
    org_class_names = org_class_names[np.in1d(org_class_names, class_names)]

    random.seed(9)

    batch_no = -1
    while batch_no<((total_classes-initial_no_of_classes)//new_classes_per_batch):
        if batch_no<0:
            images_needed_in_batch = initial_batch_size
            no_of_known_images = images_needed_in_batch
            no_classes_in_current_batch = initial_no_of_classes
        else:
            images_needed_in_batch = known_sample_per_batch + unknown_sample_per_batch
            no_of_known_images = known_sample_per_batch #int((1 - unknowns_percentage) * images_needed_in_batch)
            no_classes_in_current_batch = initial_no_of_classes + (batch_no * new_classes_per_batch)
        known_classes_of_interest = class_names[:no_classes_in_current_batch]
        unknown_classes_of_interest = class_names[no_classes_in_current_batch:no_classes_in_current_batch+new_classes_per_batch]
        known_classes = np.in1d(org_class_names, known_classes_of_interest)
        known_classes = np.arange(org_class_names.shape[0])[known_classes]
        unknown_classes = np.in1d(org_class_names, unknown_classes_of_interest)
        unknown_classes = np.arange(org_class_names.shape[0])[unknown_classes]
        known_images_to_add = np.random.choice(known_classes, no_of_known_images, replace=False)
        unknown_images_to_add = np.random.choice(unknown_classes, images_needed_in_batch-no_of_known_images, replace=False)

        knowns = list(itertools.zip_longest([batch_no+1],
                                       image_names[known_images_to_add].tolist(),
                                       org_class_names[known_images_to_add].tolist(),fillvalue=batch_no+1))

        unknowns = []
        if len(image_names[unknown_images_to_add].tolist())>0:
            unknowns = list(itertools.zip_longest(
                                [batch_no+1],
                                image_names[unknown_images_to_add].tolist(),
                                org_class_names[unknown_images_to_add].tolist(), fillvalue=batch_no+1))

        all_batches.extend(knowns)
        if batch_no>=0:
            all_batches.extend(unknowns)
        logger.info(f"Created batch no {batch_no+1} with {len(knowns)} number of knowns samples and "
                    f"{len(unknowns)} number of unknowns samples")
        batch_no+=1
    return all_batches


@vastlogger.time_recorder
@protocol
def ImageNetIncremental(initial_no_of_classes=50,
                        new_classes_per_batch=5, #10 #25
                        total_classes=100,
                        all_images=[]):
    all_batches=[]
    class_names, image_names = zip(*all_images)
    class_names = sorted(list(set(class_names)))

    np.random.seed(1993)
    class_names = np.random.choice(class_names, total_classes, replace=False)

    for batch_no, no_classes_in_current_batch in enumerate(range(initial_no_of_classes,
                                                                 len(class_names)+1,
                                                                 new_classes_per_batch)):
        classes_of_interest = class_names[:no_classes_in_current_batch]
        images_of_interest = []
        for cls_name, image in all_images:
            if cls_name in classes_of_interest:
                images_of_interest.append((batch_no, image, cls_name))
        all_batches.extend(images_of_interest)
        logger.info(f"Created batch no {batch_no} with {len(images_of_interest)} number of samples")
    return all_batches


@vastlogger.time_recorder
@protocol
def OpenWorldValidation(classes=[],
                        all_images=[]):
    classes_of_interest = classes
    images_of_interest = []
    for cls_name, image in all_images:
        if cls_name in classes_of_interest:
            images_of_interest.append((0, image, cls_name))
    return images_of_interest


if __name__ == "__main__":
    b,i,c=ImageNetIncremental(files_to_add=['imagenet_1000_val'],
                        initial_no_of_classes=5,
                        new_classes_per_batch=1,
                        total_classes=10)
    print(np.array(b))
    print(np.array(i))
    print(np.array(c))