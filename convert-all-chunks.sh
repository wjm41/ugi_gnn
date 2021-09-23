#!/bin/bash
#
# Prepare all transformations
#

#BSUB -P "testing"
#BSUB -J "convert[1-6950]"
##BSUB -J "convert[1]"
#BSUB -n 1
#BSUB -R rusage[mem=4]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -W  00:30
#BSUB -o docked/out_%I.stdout 
#BSUB -eo docked/out_%I.stderr
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
echo python convert-chunk.py --chunkprefix ugis --input output --output docked --index $CHUNK
python convert-chunk.py --chunkprefix ugis --input output --output docked --index $CHUNK
date

