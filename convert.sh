#!/bin/bash

#add tool directory to path
PATH=/data/work/dbiglari/machine_learning/c/github/bloom_c/vzgpt-bloom:$PATH

# clear the csv cache...ugh
rm -rf ~/.cache/huggingface/datasets/csv
rm ~/.cache/huggingface/datasets/*datasets_csv_*

cd /tmp/bloom-560m
rm -rf data
mkdir data
cd data
unzip -o ../pytorch_model.bin
cd archive
cd data
mkdir ../data_new
find . -mindepth 1 -exec fp32_to_fp16 {} ../data_new/{} \;
cd ..
rm -rf data
mv data_new data
cd ../..
mv pytorch_model.bin pytorch_model.bin.old
cd data
zip -r ../pytorch_model.bin archive
cd ..
rm -rf data
# keep the old data file for next training set
#rm pytorch_model.bin.old

