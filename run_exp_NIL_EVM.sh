#features_path_prefix="/net/pepper/scratch/adhamija/FeaturesCopy/OpenSetAlgos"
parent_output_dir="/tmp/adhamija/"
#parent_output_dir="/home/tahmad/work/ICCV2021/temp_log_april19_FFIL/"
features_path_prefix="/scratch/adhamija/FeaturesCopy/OpenSetAlgos"
no_of_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Each combination is feature file, distance multiplier and cover threshold
experiments_combinations=("10 moco_v1_200ep_pretrain.pth 0.6 0.8" \
                          "10 SimCLR_1x 0.7 0.8" \
                          "10 deepclusterv2 0.8 0.8" \
                          "10 moco_v2_800ep_pretrain 0.6 0.3" \
                          "10 selav2 0.6 0.8" \
                          "10 SWAV 0.7 0.8")
all_running_PIDS=()
exp_no=0
flag=-1

for exp_comb in "${experiments_combinations[@]}"; do
  read -r new_classes_per_batch feature_file DM CT <<<$exp_comb
  echo -e "\tFeatures $feature_file\t DM $DM\t CT $CT"
  port_no=$((5400+$exp_no*100))
  output_dir="${parent_output_dir}/${new_classes_per_batch}/${feature_file}/"
  mkdir -p $output_dir
  set -o xtrace
  PID=$(nohup sh -c "CUDA_VISIBLE_DEVICES=$exp_no python NIL.py \
  --training_feature_files $features_path_prefix/$feature_file/resnet50/imagenet_1000_train.hdf5 \
  --validation_feature_files $features_path_prefix/$feature_file/resnet50/imagenet_1000_val.hdf5 \
  --layer_names avgpool \
  --initialization_classes 50 --total_no_of_classes 100 --new_classes_per_batch $new_classes_per_batch --no_of_exemplars 20 \
  --output_dir $output_dir \
  --OOD_Algo EVM --tailsize 1. --distance_metric euclidean --distance_multiplier $DM --cover_threshold $CT \
  --port_no $port_no -v" >$output_dir"/${feature_file}_nohup.log" 2>&1 & echo $!)
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
  sleep 5s
  ((exp_no+=1))
done

echo "Started all processes ... waiting for them to finish"
for PID in "${all_running_PIDS[@]}"; do
  echo "Waiting for PID $PID"
  tail --pid=$PID -f /dev/null
  echo "PID $PID Ended"
done

for exp_comb in "${experiments_combinations[@]}"; do
  read -r new_classes_per_batch feature_file DM CT <<<$exp_comb
  output_dir="${parent_output_dir}/${new_classes_per_batch}/${feature_file}/"
  mv $output_dir/log.txt $output_dir/${feature_file}_log.txt
done