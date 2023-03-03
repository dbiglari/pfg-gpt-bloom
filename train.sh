#!/bin/bash

# load up python environment for training
source ~/conda_env.sh ./anacondaenv.txt

if [[ $2 == bigscience* ]]
then
  echo pulling from bigscience.
else

  if [ -f $2/pytorch_model.bin.old ]
  then
    rm $2/pytorch_model.bin
    mv $2/pytorch_model.bin.old $2/pytorch_model.bin
  fi
fi


PYTORCH_RUN_CLM_PATH=/home/silicon-admin/eclipse-python-workspace/Bloom_Training/src/examples/pytorch/language-modeling
TRAINING_FILE_PATH=$(realpath $1)

echo use jiZ40CYXPMQy6RRRmOldMsn9kylOv5ER env

START_DIR=$(pwd)


cd $PYTORCH_RUN_CLM_PATH
python ./run_clm.py --model_name_or_path $2 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 --train_file $TRAINING_FILE_PATH --do_train   --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --gradient_accumulation_steps 4   --learning_rate 3e-5   --num_train_epochs 1   --output_dir /tmp/bloom-560m   --overwrite_output_dir  --no_cuda
cd $START_DIR


# change 32 bit weights to 16 bit weights to save disk space
./convert.sh

