#!/bin/bash

source /homes/deng106/mc.defs 

mpiicpc main.cpp -o spike_like -openmp 
mpirun -n 4 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm  -genv OMP_NUM_THREADS 8 ./spike_like ../hw2a/L_hat.mtx ../hw2a/f.mtx  y.mtx
rm spike_like


