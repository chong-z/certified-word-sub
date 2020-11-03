#!/bin/bash

for s in 42 123 1234567 100007 17323; do
    FF="--random-seed $s -d 100 --pool mean --dropout-prob 0.2 -b 32 --data-cache-dir .cache --save-best-only --enable-wandb --wandb-tag Oct10TrainJiaCertifiedModels"

    python src/train.py classification bow --out-dir model_data/bow_normal_$s -T 60 $FF
    python src/train.py classification bow --out-dir model_data/bow_cert_$s -T 60 --full-train-epochs 20 -c 0.8 $FF

    python src/train.py classification cnn --out-dir model_data/cnn_normal_$s -T 60 $FF
    python src/train.py classification cnn --out-dir model_data/cnn_cert_$s -T 60 --full-train-epochs 20 -c 0.8 $FF

    python src/train.py classification lstm --out-dir model_data/lstm_normal_$s -T 30 $FF
    python src/train.py classification lstm --out-dir model_data/lstm_cert_$s -T 30 --full-train-epochs 10 -c 0.8 $FF
done || exit 1
