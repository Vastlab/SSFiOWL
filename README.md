### Self-Supervised Features Improve Open-World Learning

This repo reproduces results from the paper "Self-Supervised Features Improve Open-World Learning"

Please use the following BibTex to cite this work.
```
article{dhamija2021self,
  title={Self-Supervised Features Improve Open-World Learning},
  author={Dhamija, Akshay Raj and Ahmad, Touqeer and Schwan, Jonathan and Jafarzadeh, Mohsen and Li, Chunchun and Boult, Terrance E},
  journal={arXiv preprint arXiv:2102.07848},
  year={2021}
}
```
###### Dependencies
This repo is dependent on the repo https://github.com/Vastlab/vast that contains some useful functionality for various projects at VastLab.
Please install it using `pip install git+https://github.com/Vastlab/vast.git`.

###### Feature Extraction
We pre-extract features from the self supervised networks and use them for all our experiments.
For extracting the features please use either `FromCSV.py` or `FromDirectoryStructures.py` based on how your data is structured.
The scripts are present at https://github.com/Vastlab/vast/tree/main/vast/scripts/FeatureExtractors.

###### Non-Backpropagating Incremental Learning (NIL)
Sample command used to run incremental learning experiments using the NIL approach

```
time python NIL.py --training_feature_files {Feature_Path}/resnet50/imagenet_1000_train.hdf5 \
 --validation_feature_files {Feature_Path}/resnet50/imagenet_1000_val.hdf5 \
 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean \
 --initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 \ 
 --output_dir /tmp/ --distance_multiplier 0.7 --no_of_exemplars 20
```

###### Non-backpropagting Open World Learning (NOWL)
Sample command used to run open world learning experiments using the NOWL approach

```
time python NOWL.py --training_feature_files {Feature_Path}/resnet50/imagenet_1000_train.hdf5 \
 --validation_feature_files {Feature_Path}/resnet50/imagenet_1000_val.hdf5 \
 --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean \
 --initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 \
 --output_dir /tmp/ --distance_multiplier 0.7 --no_of_exemplars 20 --cover_threshold 0.7 \
 --known_sample_per_batch 2500 --unknown_sample_per_batch 2500 --initial_no_of_samples 15000
```
