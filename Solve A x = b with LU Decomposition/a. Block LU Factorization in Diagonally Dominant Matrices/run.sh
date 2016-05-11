source /homes/deng106/mc.defs
cc=icc
MPICC=mpiicc

MKL_MIC_ENABLE=1

MKL="/p/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group /p/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_intel_ilp64.a /p/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_core.a /p/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_intel_thread.a /p/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm"

LIB="-DMKL_ILP64 -qopenmp -I/p/intel/compilers_and_libraries_2016.1.150/linux/mkl/include"

echo "MKL loading..."
$MPICC $LIB -g main.cpp -o main $MKL


mpirun -n 8 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm  -genv OMP_NUM_THREADS 8 ./main A_dd.mtx b_dd.mtx x.mtx
#mpirun -n 8 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm  -genv OMP_NUM_THREADS 8 ./main tinyA_dd.mtx tinyb_dd.mtx x.mtx
#mpirun -n 8 -f ./mpd.hosts -perhost 1 -genv I_MPI_DEVICE=ssm  -genv OMP_NUM_THREADS 8 ./main tinyA_ndd.mtx tinyb_ndd.mtx x.mtx

rm main
