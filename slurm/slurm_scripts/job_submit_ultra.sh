#! /bin/bash
model_dir=/rds-d2/user/wjm41/hpc-work/models
script_dir=/home/wjm41/symlinks/rds/datasets/Ugis/scripts

for dataset in AmpC
do
for optimizer in Adam
do
echo Training model for ${dataset} with ${optimizer}
save_name=${model_dir}/${dataset}_${optimizer}
log_dir=/home/wjm41/symlinks/rds/datasets/Ugis/scripts/runs/ultra-large/${dataset}/${optimizer}

sbatch --job-name=${dataset}_${optimizer} \
       --output=${script_dir}/slurm_logs/subm_mpnn_ultra/${dataset}_${optimizer}.out \
       --export=dataset=${dataset},optimizer=${optimizer},save_name=${save_name},log_dir=${log_dir} subm_mpnn_ultra
done
done