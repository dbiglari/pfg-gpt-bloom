#!/bin/bash

./pfg-gpt-bloom -f prompts/prompt.txt -T 12 -m greedy -M bloom-560m -F /etc/pfg_gpt_config
