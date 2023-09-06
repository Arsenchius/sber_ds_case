#!/bin/bash

TRAIN_DATA=/home/kenny/sber_ds_case/data/raw/test.csv
OUTPUT=/home/kenny/sber_ds_case/data/processed
STOPWORDS=/home/kenny/sber_ds_case/data/raw/stopwords.txt

EXECUTABLE=$1
CHUNK_SIZE=$2

python $EXECUTABLE --train-data-path $TRAIN_DATA --output-dir-path $OUTPUT --chunk-size $CHUNK_SIZE --path-to-stopwords $STOPWORDS
