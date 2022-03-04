#! /bin/bash
model_dir=/rds-d2/user/wjm41/hpc-work/models
script_dir=/home/wjm41/symlinks/rds/datasets/Ugis/scripts

for optimizer in FelixExpHD
do
echo ${optimizer}
save_name=${model_dir}/ugi_full_${optimizer}
log_dir=/home/wjm41/symlinks/rds/datasets/Ugis/scripts/runs/optimisation_comparison/${optimizer}/

sbatch --job-name=ugi_taut_${optimizer} \
       --output=${script_dir}/slurm_logs/subm_mpnn_opt/${optimizer}_opt_full.out \
       --export=optimizer=${optimizer},save_name=${save_name},log_dir=${log_dir} subm_mpnn_opt
done