#!/bin/bash


for eps in $(cat ../gen_epsilons_pb.txt)
	do
		echo $eps
		Rscript alc_model.R $eps 100
	done
