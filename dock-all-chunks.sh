#!/bin/bash
#
# Prepare all transformations
#

#BSUB -P "testing"
##BSUB -J "ugis[1-6950]"
#BSUB -J "ugis[1787,2889]"
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -W  24:00
#BSUB -o output/out_%I.stdout 
#BSUB -eo output/out_%I.stderr
##BSUB -cwd "/scratch/%U/%J"
#BSUB -L /bin/bash

# quit on first error
#set -e

source ~/.bashrc
export NUMEXPR_MAX_THREADS=1

cd $LS_SUBCWD

conda activate perses-fah-0.8.1

# Launch my program.
export CHUNK=$(expr $LSB_JOBINDEX - 1) # CHUNK is zero-indexed
echo $CHUNK

date
echo python dock-chunk.py --chunk $CHUNK --chunksize 2500 --output output/ugis
python dock-chunk.py --chunk $CHUNK --chunksize 2500 --output output/ugis
date
