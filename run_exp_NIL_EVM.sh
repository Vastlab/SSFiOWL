time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.3
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_MOCO_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_SimCLR_1x_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_MOCO_v1_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_sela_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_deepcluster_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 10 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_10_class/NIL_self-supervised_SWAV_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt







time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.6
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_MOCO_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_SimCLR_1x_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_MOCO_v1_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.6  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_sela_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_deepcluster_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 5 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_5_class/NIL_self-supervised_SWAV_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt








time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.7
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_MOCO_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_SimCLR_1x_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_MOCO_v1_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_sela_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_deepcluster_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 2 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_2_class/NIL_self-supervised_SWAV_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt









time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v2_800ep_pretrain/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.7  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_MOCO_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SimCLR_1x/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_SimCLR_1x_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/moco_v1_200ep_pretrain.pth/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.7
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_MOCO_v1_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/selav2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_sela_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/deepclusterv2/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_deepcluster_v2_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt


time python NIL.py --training_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_train.hdf5 \
--validation_feature_files /scratch/adhamija/FeaturesCopy/OpenSetAlgos/SWAV/resnet50/imagenet_1000_val.hdf5 --layer_names avgpool \
--initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch 1 --output_dir /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ --no_of_exemplars 20 \
--OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier 0.8  --cover_threshold 0.8
cp /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/ResNet50_1_class/NIL_self-supervised_SWAV_log.txt
rm /home/tahmad/work/ICCV2021/temp_log_april19_FFIL/log.txt

