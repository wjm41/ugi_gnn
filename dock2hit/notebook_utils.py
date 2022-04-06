from typing import List


def write_slurm_script(job_name: str,
                       run_time: str,
                       output_name: str,
                       package_dir: str,
                       script: str,
                       args: List,
                       file_name: str,
                       email: bool = False):
    slurm_options = [
        '#!/bin/bash',
        f'#SBATCH -J {job_name}',
        '#SBATCH -A LEE-WJM41-SL2-GPU',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        '#SBATCH --gres=gpu:1',
        f'#SBATCH --time={run_time}',
        '#SBATCH --mail-user=wjm41@cam.ac.uk',
        f'#SBATCH --output={output_name}',
        '#SBATCH -p ampere',
    ]
    if email:
        slurm_options.append('#SBATCH --mail-type=ALL')

    module_options = [
        '. /etc/profile.d/modules.sh',
        'module purge',
        'module load rhel8/default-amp',
        'module load miniconda/3',
        'source activate dgl_life',
    ]

    pre_empt = f'cd {package_dir}; pip install . --use-feature=in-tree-build'

    slurm_options = '\n'.join(slurm_options)
    module_options = '\n'.join(module_options)
    command_to_run = ' '.join([script]+args)

    string_to_write = f'{slurm_options}\n{module_options}\n{pre_empt}\n{command_to_run}'

    with open(file_name, 'w') as f:
        f.write(string_to_write)

    return
