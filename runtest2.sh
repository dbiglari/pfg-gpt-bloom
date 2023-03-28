#!/bin/bash

#(sleep 1 && sudo chrt -p 99 $(pidof pfg-gpt-bloom) && sudo renice -n -20 -p $(pidof pfg-gpt-bloom))&


prompt=$1
shift
modelname=$1
shift
tokenmode=$1
shift
threads=$1
shift
useopencl=$1
shift
time ./pfg-gpt-bloom -f $prompt -t 1.0 -z 0.0 -m $tokenmode -T $threads -F /etc/pfg_gpt_config -M $modelname -C $useopencl $@
