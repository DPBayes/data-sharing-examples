#!/bin/bash

for seed in $(cat gen_seeds.txt)
	do
		Rscript alc_model.R 0.74 10 $seed &
		Rscript alc_model.R 1.99 10 $seed &
		Rscript alc_model.R 3.92 10 $seed &
		wait
	done
