#source /homes/deng106/mc.defs
CC = icc
MPICC=mpiicc

MKL_MIC_ENABLE=4

MKL =    ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm

FLAGS=  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include
all: main
main: main.cpp
	$(MPICC) $(FLAGS) main.cpp -o main $(MKL)
	./main testA.mtx testa.mtx x.mtx

clean:
	rm -f main *.o

 
