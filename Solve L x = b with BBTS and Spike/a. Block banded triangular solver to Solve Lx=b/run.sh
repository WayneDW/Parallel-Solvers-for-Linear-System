
export  OMP_NUM_THREADS=$1
source ../../mc.defs
echo "Current running core number is: " $OMP_NUM_THREADS


path="/homes/deng106/hw/hw2a/"
icc -openmp block_banded_triangular_v4.c mmio.c matcomput.c -o ./BBTS
./BBTS $path/L.mtx $path/b.mtx $path/x.mtx

