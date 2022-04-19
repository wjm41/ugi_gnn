from typing import List
import logging
from math import log, floor
import subprocess

import pandas as pd

from torch.cuda import is_available as cuda_is_available
from torch.cuda import get_device_name as cuda_get_device_name


def human_len(input, byte=False):
    """Given a number or list like, returns the size/length of the object in human-readable form

    As an example, human_len(2048, byte=True) = 2KB ; human_len(np.ones(14000)) = 14K;

    Args:
        input: A number or an object with the __len__ method.
        byte: Whether or not to treat the length in byte form.

    Returns:
        length_string (string): Human-readable string of the object size.
    """
    try:
        input = input.__len__()
    except AttributeError:
        pass

    if byte:
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        k = 1024.0
    else:
        units = ['', 'K', 'M', 'G', 'T', 'P']
        k = 1000.0
    magnitude = int(floor(log(input, k)))
    length_string = '%.2f%s' % (input / k**magnitude, units[magnitude])
    return length_string


def bash_command(cmd: str):
    """Utility script for running a bash command from a python file.

    Args:
        cmd (str): bash command to-be-run.
    """
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.communicate()


def get_device() -> str:
    """Detects CUDA availability and returns the appropriate torch device.

    Returns:
        device (str): device to-be used in torch
    """
    if cuda_is_available():
        logging.info(f'using GPU: {cuda_get_device_name()}')
        device = 'cuda'
    else:
        logging.info('No GPU found, using CPU')
        device = 'cpu'
    return device


def write_slurm_script(job_name: str,
                       run_time: str,
                       output_name: str,
                       package_dir: str,
                       script: str,
                       args: List,
                       file_name: str,
                       email: bool = False,
                       gpu: bool = False):

    if gpu:
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
    else:
        slurm_options = [
            '#!/bin/bash',
            f'#SBATCH -J {job_name}',
            '#SBATCH -A LEE-WJM41-SL2-CPU',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            ' #SBATCH --cpus-per-task=1',
            f'#SBATCH --time={run_time}',
            '#SBATCH --mail-user=wjm41@cam.ac.uk',
            f'#SBATCH --output={output_name}',
            '#SBATCH -p icelake-himem',
        ]
    if email:
        slurm_options.append('#SBATCH --mail-type=ALL')

    if gpu:
        module_options = [
            '. /etc/profile.d/modules.sh',
            'module purge',
            'module load rhel8/default-amp',
            'module load miniconda/3',
            'source activate dgl_life',
        ]
    else:
        module_options = [
            '. /etc/profile.d/modules.sh',
            'module purge',
            'module load rhel8/default-amp',
            'module load miniconda/3',
            'source activate dgl_cpu',
        ]

    pre_empt = f'cd {package_dir}; pip install . --use-feature=in-tree-build'

    slurm_options = '\n'.join(slurm_options)
    module_options = '\n'.join(module_options)
    command_to_run = ' '.join([script]+args)

    string_to_write = f'{slurm_options}\n{module_options}\n{pre_empt}\n{command_to_run}'

    with open(file_name, 'w') as f:
        f.write(string_to_write)

    return


def read_csv_or_pkl(file_path):

    if file_path.split('.')[-1] == 'pkl':
        df = pd.read_pickle(file_path).reset_index()
    elif file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path).reset_index(drop=True)
    else:
        raise ValueError('Unrecognised file suffix!')
    return df
