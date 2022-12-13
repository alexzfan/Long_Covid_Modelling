#!/bin/bash

echo GRID SEARCH HYPERPARAMS 
for lr in 0.01 0.001 0.0001 
do
    for l2_wd in 0.01 0.001 0.0001
    do
        for dropout_prob in 0.2 0.5 0.8
        do
            for hidden_size in 25 50 100
            do
            python train.py --lr $lr --l2_wd $l2_wd --drop_prob $dropout_prob --hidden_size $hidden_size -n base_ff_v2.lr-${lr}.l2_wd-${l2_wd}.drop_prob-${dropout_prob}.hidden-${hidden_size}
            done
        done
    done
done