#!/bin/bash


for eps in $(cat ../gen_epsilons.txt)
	do
	echo $eps
	for seed in $(cat ../gen_seeds.txt)
		do
			Rscript alc_model.R $eps 10 $seed
		done
	done
