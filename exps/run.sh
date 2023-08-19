#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
python fedhald_main.py --dataset mnist --num_classes 10 --num_users 20 --ways 5 --shots 224 --optimizer sgd --local_bs 32 --stdev 2 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20

python fedhald_main.py --dataset cifar10 --num_classes 10 --num_users 20 --ways 5 --shots 500 --optimizer sgd --local_bs 32 --stdev 2 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20

python fedhald_main.py --dataset cifar100 --num_classes 100 --num_users 20 --ways 50 --shots 352 --optimizer sgd --local_bs 32 --stdev 20 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20