#!/bin/bash

for mon in 1 2 3 4 5 6 7 8 9 10 11 12
#for mon in 5 6 7 8 9 11 12
do
	for lead in 1 2 3 4 5 6 7 8 9 10 11 12
	#for lead in  11 12
	do
      export NSTEP=$lead
      export  NMON=$mon

      python CNN_CMIP6_ens43_LeNet-5_tf2_v1.5.py



	done
done
