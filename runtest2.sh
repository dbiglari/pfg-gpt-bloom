#!/bin/bash



./pfg-gpt-bloom -f $1 -t 1.0 -z 0.0 -m sampling -T 12 -F /etc/pfg_gpt_config -M $2
