import itertools
import sys
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # removes annoying RDKit warnings

# from mpi4py import MPI

file_dir = sys.argv[1]

# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()
# mpi_size = mpi_comm.Get_size()

mpi_rank = int(sys.argv[2])
mpi_size = int(sys.argv[3])

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

df = pd.read_csv(file_dir+'/ugis_processed_new.csv')

low, high = return_borders(mpi_rank, len(df), mpi_size)

enumerator = rdMolStandardize.TautomerEnumerator()

def return_taut(smi):
    canon = enumerator.Canonicalize(MolFromSmiles(smi))
    return MolToSmiles(canon)

df = df.iloc[low:high]

tqdm.pandas(smoothing=0, miniters=100) # erractic progress, use full average to smooth out 
df['smiles'] = df['smiles'].progress_apply(return_taut)

df.to_csv(file_dir+'/scripts/tmp/tautcan_'+str(mpi_rank)+'.csv', index=False)

# # wait for all processes to finish
# mpi_comm.Barrier()

# if mpi_rank==0:
#     for n in range(1, mpi_size):
#         df_read = pd.read_csv(file_dir+'/tmp/tautcan_'+str(n)+'.csv')
#         df = pd.concat([df, df_read])
#     df.to_csv(file_dir+'/tautcan_lib.csv', index=False)
