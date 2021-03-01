#!/bin/bash

if [ "$1" = "female" ]; then
	for seed in $(cat gen_seeds.txt)
		do
			#Rscript alc_model_female.R 0.74 10 $seed &
			Rscript alc_model_female.R 1.99 10 $seed &
			Rscript alc_model_female.R 3.92 10 $seed &
			wait
		done
else
	for seed in $(cat gen_seeds.txt)
		do
			Rscript alc_model_male.R 0.74 10 $seed &
			#Rscript alc_model_male.R 1.99 10 $seed &
			#Rscript alc_model_male.R 3.92 10 $seed &
			wait
		done
fi
