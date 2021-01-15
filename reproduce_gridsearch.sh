features_path_prefix="/net/pepper/scratch/adhamija/FeaturesCopy/OpenSetAlgos"
#features_path_prefix="/scratch/adhamija/FeaturesCopy/OpenSetAlgos"
no_of_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
feature_files=('moco_v1_200ep_pretrain.pth' 'SimCLR_1x' 'deepclusterv2' 'moco_v2_800ep_pretrain' 'selav2' 'SWAV')
all_running_PIDS=()
exp_no=0
flag=-1

DM=($(seq 0.1 0.10 1.1))
CT=($(seq 0.1 0.10 0.9))
DM=$(printf "%s " "${DM[@]}")
CT=$(printf "%s " "${CT[@]}")

for feature_file in "${feature_files[@]}"; do
  echo -e "Features\t $feature_file"
  port_no=$((5400+$exp_no*100))
  output_dir="/scratch/adhamija/SSFiOWL/Grid_Search_Results/${feature_file}/"
  mkdir -p $output_dir
  set -o xtrace
  PID=$(nohup sh -c "CUDA_VISIBLE_DEVICES=$exp_no python NIL.py \
  --training_feature_files $features_path_prefix/$feature_file/resnet50/imagenet_1000_train.hdf5 \
  --validation_feature_files $features_path_prefix/$feature_file/resnet50/imagenet_1000_val.hdf5 \
  --layer_names avgpool --OOD_Algo EVM --tailsize 1. --distance_metric euclidean \
  --initialization_classes 30 --total_no_of_classes 50 \
  --distance_multiplier $DM --cover_threshold $CT \
  --output_dir $output_dir --no_of_exemplars 20 --port_no $port_no -vvv" >/dev/null 2>&1 & echo $!)
  set +o xtrace
  echo "Started PID $PID"
  all_running_PIDS+=($PID)
  if [ ${#all_running_PIDS[@]} -eq $no_of_gpus ]
  then
    echo "Waiting for PID ${all_running_PIDS[0]}"
    tail --pid=${all_running_PIDS[0]} -f /dev/null
    echo "PID Ended ${all_running_PIDS[0]}"
    all_running_PIDS=("${all_running_PIDS[@]:1}") #removed the 1st element
    exp_no=0+$flag
    ((flag+=1))
    if [ $flag -eq $no_of_gpus ]
    then
      flag=0
    fi
  fi
  echo "Waiting a moment before starting next set of experiments"
  sleep 30s
  ((exp_no+=1))
done