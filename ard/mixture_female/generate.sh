#!/bin/bash

source activate myenv

for eps in $(cat gen_epsilons.txt)
	do
	echo $eps
	for seed in $(cat gen_seeds.txt)
		do
			python3 generate_data_by_seed.py $eps $seed
		done
	done
