#!/bin/bash

python src/train.py classification cnn model_data/cnn_normal --data-cache-dir .cache -d 100 --pool mean -T 10 --dropout-prob 0.2 -b 32 --save-best-only
python src/train.py classification cnn model_data/cnn_cert --data-cache-dir .cache -d 100 --pool mean -T 60 --full-train-epochs 20 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only
python src/train.py classification cnn model_data/cnn_c05 --data-cache-dir .cache -d 100 --pool mean -T 60 --full-train-epochs 20 -c 0.5 --dropout-prob 0.2 -b 32 --save-best-only


python src/train.py classification bow model_data/bow_normal -d 100 --pool mean -T 60 --dropout-prob 0.2 -b 32 --save-best-only
python src/train.py classification bow model_data/bow_cert -d 100 --pool mean -T 60 --dropout-prob 0.2 --full-train-epochs 20 -c 0.8 -b 32 --save-best-only


python src/train.py classification lstm model_data/lstm_normal -d 100 --pool mean -T 30 --dropout-prob 0.2 -b 32 --save-best-only
python src/train.py classification lstm model_data/lstm_cert -d 100 --pool mean -T 30 --dropout-prob 0.2 --full-train-epochs 10 -c 0.8 -b 32 --save-best-only
