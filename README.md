### Self-Supervised Features Improve Open-World Learning

This repo reproduces results from the paper "Self-Supervised Features Improve Open-World Learning"

Please use the following BibTex to cite this work.
```
@article{dhamija2020SSFIOWL,
  title={Self-Supervised Features Improve Open-World Learning},
  year={2020}
}
```
###### Dependencies
This repo is dependent on the repo https://github.com/Vastlab/utile that contains some useful functionality for various projects at VastLab.
Please ensure this repo is cloned and in your path.

###### Feature Extraction
Since none of our approaches perform any backpropagation, we pre-extract features from the self supervised networks and use them for all our experiments.
For extracting the features please use the script provided at https://github.com/Vastlab/utile/blob/master/tools/FeatureExtraction.py 

###### Non-Backpropagating Incremental Learning (NIL)
Sample command used to run incremental learning experiments using the NIL approach

```
time python NIL.py --training_feature_files {Feature_Path}/resnet50/imagenet_1000_train.hdf5 --validation_feature_files {Feature_Path}/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean --initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /tmp/ --distance_multiplier 0.7 --no_of_exemplars 20
```

NIL Grid-Search Results:


|     Features    | New Classes | Distance Multiplier | Cover Threshold | Accuracy |
|:---------------:|:-----------:|:-------------------:|:---------------:|:--------:|
|       SWAV      |      2      |         0.6         |       0.8       |   89.17  |
|       SWAV      |      5      |         0.7         |       0.8       |   92.14  |
|       SWAV      |      10     |         0.7         |       0.8       |   90.85  |
|      Selav2     |      2      |         0.6         |       0.8       |   88.64  |
|      Selav2     |      5      |         0.6         |       0.8       |   91.92  |
|      Selav2     |      10     |         0.6         |       0.8       |   89.66  |
|      SIMCLR     |      2      |         0.6         |       0.8       |   86.00  |
|      SIMCLR     |      5      |         0.7         |       0.8       |   87.12  |
|      SIMCLR     |      10     |         0.7         |       0.8       |   85.54  |
|     MoCo V1     |      2      |         0.6         |       0.8       |   87.35  |
|     MoCo V1     |      5      |         0.6         |       0.8       |   88.56  |
|     MoCo V1     |      10     |         0.6         |       0.8       |   85.64  |
|     MoCo V2     |      2      |         0.5         |       0.8       |   89.73  |
|     MoCo V2     |      5      |         0.5         |       0.8       |   91.59  |
|     MoCo V2     |      10     |         0.6         |       0.8       |   90.43  |
| Deep Cluster V2 |      2      |         0.7         |       0.8       |   89.10  |
| Deep Cluster V2 |      5      |         0.7         |       0.8       |   91.51  |
| Deep Cluster V2 |      10     |         0.7         |       0.8       |   89.22  |


###### Non-backpropagting Open World Learning (NOWL)
Sample command used to run open world learning experiments using the NOWL approach

```
time python NOWL.py --training_feature_files /scratch/adhamija/FeaturesCopy/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_train.hdf5 --validation_feature_files /scratch/adhamija/FeaturesCopy/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean --initialization_classes 50 --total_no_of_classes 100 --output_dir /tmp/ --distance_multiplier 0.7 --no_of_exemplars 20 --new_classes_per_batch 10 --cover_threshold 0.7 --port_no 3393 --known_sample_per_batch 2500 --unknown_sample_per_batch 2500 --initial_no_of_samples 15000
```
