# SOWL

Sample command used to run incremental learning experiments

```
time python main.py --training_feature_files /scratch/adhamija/FeaturesCopy/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 --validation_feature_files /scratch/adhamija/FeaturesCopy/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean --Clustering_Algo finch --initialization_classes 500 --total_no_of_classes 1000 --new_classes_per_batch 100 --output_dir /tmp/
```

Sample command used to run open world learning experiments

```
time python open_world.py --training_feature_files /scratch/adhamija/FeaturesCopy/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 --validation_feature_files /scratch/adhamija/FeaturesCopy/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean --Clustering_Algo finch --initialization_classes 50 --total_no_of_classes 500 --new_classes_per_batch 10 --output_dir /tmp/ --distance_multiplier 0.7 --accumulation_algo learn_new_unknowns --known_sample_per_batch 2500 --unknown_sample_per_batch 2500 --initial_no_of_samples 15000 --no_of_exemplars 20
```
