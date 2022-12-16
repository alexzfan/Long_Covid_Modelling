#!/bin/bash

echo GRID SEARCH HYPERPARAMS SAMPLING
for num_samples in 2 4 8
do
    for lr in 0.01 0.001 0.0001 
    do
        for l2_wd in 0.01 0.001 0.0001
        do
            for dropout_prob in 0.2 0.5 0.8
            do
                for hidden_size in 25 50 100
                do
                python train_subsamples.py --num_samples $num_samples --lr $lr --l2_wd $l2_wd --drop_prob $dropout_prob --hidden_size $hidden_size -n subsampled_ff_40_epoch.num_samples-${num_samples}.lr-${lr}.l2_wd-${l2_wd}.drop_prob-${dropout_prob}.hidden-${hidden_size} --num_epochs 40
                done
            done
        done
    done
done