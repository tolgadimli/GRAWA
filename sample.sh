#!/bin/bash 

# -------- Experiment Setup --------
DATE=`date +%Y-%m-%d`
dataset='cifar10'; model='resnet20'; datadir='./dataset'
m=0.9; batch_size=128; minutes=15; wd=0.0001
lr=0.1
# -------- Required Info --------
num_groups=1 # change it to >1 if more nodes are present
if [[ $num_groups -eq 1 ]]
then
  cur_group=0; ip_addr=current.node.ip.address
else
  cur_group=$1; ip_addr=central.node.ip.address
fi

dist_op='DataParallel'
opt='SGD'
check_dir="DataParallel-$opt-$dataset-$model-w$num_groups-$cur_group-$DATE"
for seed in 36 720 1042; do
  exp_name="DataParallel-lr-$lr-m-$m-b-$batch_size-comm=$g_comm=$seed"
  echo "STARTING DataParallel-lr=$lr(comm=$g_comm)"
  python ./codes/data_parallel.py --datadir $datadir --dataset $dataset --model $model --wd $wd\
    --optimizer $opt --batch_size $batch_size --lr $lr --mom $m --dist_ip $ip_addr --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir --minutes $minutes --cpu_seed $seed --gpu_seed $seed 
  echo "FINISHING DataParallel-lr=$lr(comm=$g_comm)" 
done

dist_op='DataParallel'
opt='SAM'
check_dir="DataParallel-$opt-$dataset-$model-w$num_groups-$cur_group-$DATE"
for seed in 36 720 1042; do
  exp_name="DataParallel-lr-$lr-m-$m-b-$batch_size-comm=$g_comm=$seed"
  echo "STARTING DataParallel-lr=$lr(comm=$g_comm)"
  python ./codes/data_parallel.py --datadir $datadir --dataset $dataset --model $model --wd $wd\
    --optimizer $opt --batch_size $batch_size --lr $lr --mom $m --dist_ip $ip_addr --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir --minutes $minutes --cpu_seed $seed --gpu_seed $seed 
  echo "FINISHING DataParallel-lr=$lr(comm=$g_comm)" 
done

dist_op='EASGD'
check_dir="$dist_op-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=4; c2=0.1; p2=0
for seed in 36 720 1042; do 
  l_comm=$(expr $g_comm)
  exp_name="$dist_op-lr-$lr-comm=$g_comm=c2-$c2-m-$m-p2-$p2-$seed"
  echo "STARTING $dist_op-lr=$lr(comm=$g_comm)"
  python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
    --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
    --batch_size $batch_size --lr $lr --c2 $c2 --p2 $p2 --mom $m \
    --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes \
    --cpu_seed $seed --gpu_seed $seed 
  echo "FINISHING $dist_op-lr=$lr(comm=$g_comm)"
done

dist_op='LSGD'
check_dir="$dist_op-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=4; c2=0.1; p2=0.1
for seed in 36 720 1042; do 
  l_comm=$(expr $g_comm)
  exp_name="$dist_op-lr-$lr-comm=$g_comm=c2-$c2-m-$m-p2-$p2-$seed"
  echo "STARTING $dist_op-lr=$lr(comm=$g_comm)"
  python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
    --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
    --batch_size $batch_size --lr $lr --c2 $c2 --p2 $p2 --mom $m \
    --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes \
    --cpu_seed $seed --gpu_seed $seed
  echo "FINISHING $dist_op-lr=$lr(comm=$g_comm)"
done

dist_op='LGRAWA'
check_dir="$dist_op-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=32; c2=0.5; p2=0.01
for seed in 36 720 1042; do 
  l_comm=$(expr $g_comm)
  exp_name="$dist_op-lr-$lr-comm=$g_comm=c2-$c2-m-$m-p2-$p2-$seed"
  echo "STARTING $dist_op-lr=$lr(comm=$g_comm)"
  python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
    --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
    --batch_size $batch_size --lr $lr --c2 $c2 --p2 $p2 --mom $m \
    --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes \
    --cpu_seed $seed --gpu_seed $seed  --grawa_prox --grawa_momentum
  echo "FINISHING $dist_op-lr=$lr(comm=$g_comm)"
done


dist_op='LGRAWA'
check_dir="$dist_op-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=32; c2=0.5; p2=0.01
for seed in 36 720 1042; do 
  l_comm=$(expr $g_comm)
  exp_name="$dist_op-lr-$lr-comm=$g_comm=c2-$c2-m-$m-p2-$p2-$seed"
  echo "STARTING $dist_op-lr=$lr(comm=$g_comm)"
  python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
    --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
    --batch_size $batch_size --lr $lr --c2 $c2 --p2 $p2 --mom $m \
    --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes \
    --cpu_seed $seed --gpu_seed $seed  --grawa_prox --grawa_momentum
  echo "FINISHING $dist_op-lr=$lr(comm=$g_comm)"
done

