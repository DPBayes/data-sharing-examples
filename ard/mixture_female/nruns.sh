#!/bin/bash

for seed in $(cat seeds.txt)
	do
	echo "Seed" $seed
	for dp_sigma in $(cat sigmas.txt)
		do
			python3 female_main.py $dp_sigma 10 $seed
		done
	done
