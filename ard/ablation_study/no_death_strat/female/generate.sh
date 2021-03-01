#!/bin/bash

source activate myenv

for seed in $(cat gen_seeds.txt)
	do
		python3 generate_data_by_seed.py 0.74 $seed &
		python3 generate_data_by_seed.py 1.99 $seed &
		python3 generate_data_by_seed.py 3.92 $seed &
		wait
	done
