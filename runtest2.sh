#!/bin/bash

#(sleep 1 && sudo chrt -p 99 $(pidof pfg-gpt-bloom) && sudo renice -n -20 -p $(pidof pfg-gpt-bloom))&

time ./pfg-gpt-bloom -f $1 -t 1.0 -z 0.0 -m $3 -T $4 -F /etc/pfg_gpt_config -M $2 -C $5
