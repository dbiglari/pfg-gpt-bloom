
# e.g. DATASET=bigscience/bloom-560m
DATASET=$1
CUDA_VISIBLE_DEVICES=-1 python ./run_clm.py --model_name_or_path $DATASET --dataset_name tatsu-lab/alpaca --train_file train.csv --do_train   --per_device_train_batch_size 1 --per_device_eval_batch_size 1  --gradient_accumulation_steps 128   --learning_rate 2e-5   --num_train_epochs 1   --output_dir /tmp/bloom-560m   --overwrite_output_dir --no_cuda
