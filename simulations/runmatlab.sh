#!/bin/bash

echo Running Matlab Script

n1=10

T=1000

numc=30

for k in 2 4 8 10
do 
	for n0 in 20 40 100
	do
		start=`date +%s`
		echo k = $k, n0 = $n0
		matlab -nodisplay -nodesktop -r "Main_MonteCarlo_v2_f($T, $k, $n1, $n0, $numc);quit"
		end=`date +%s`
		echo Execution time was `expr $end - $start` seconds.
	done
done

echo Script Done