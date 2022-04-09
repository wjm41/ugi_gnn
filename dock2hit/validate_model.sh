#!/bin/bash

#python mpnn_predict.py -tp $DIR/ugis_processed.csv -p $DIR/docked2/val.csv #-p $DIR/docked2/ugis_processed.csv
#python mpnn_predict.py -tp $DIR/ugis_processed.csv -p $DIR/docked2/ugis_processed.csv -load_name $model_dir/model_epoch_1.pt
#python mpnn_predict.py -tp $DIR/ugis_processed.csv -p $DIR/docked2/val.csv 

DIR=/rds-d2/user/wjm41/hpc-work/datasets/Ugis
model_dir=/rds-d2/user/wjm41/hpc-work/models

python mpnn_batched.py -p $DIR/small_test.csv -val_path $DIR/docked2/ugis_processed.csv -n_trials 1 -batch_size 32 -save_batch 1000 -n_epochs 50000 -ts 0 -lr 1e-2 -save_name test #-load_name $model_dir/ugi_pretrained_smalllr/model_epoch_1.pt
