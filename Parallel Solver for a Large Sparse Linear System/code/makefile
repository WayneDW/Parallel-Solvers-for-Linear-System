export PETSC_DIR=/u/u24/zhang701/MC_PETSC/


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


MAIN = main
$(MAIN): $(MAIN).o rcm.o chkopts
	-${CLINKER} -o $(MAIN) $(MAIN).o rcm.o  ${PETSC_KSP_LIB}
	${RM} $(MAIN).o


MAIN0 = main_0
$(MAIN0): $(MAIN0).o rcm.o chkopts
	-${CLINKER} -o $(MAIN0) $(MAIN0).o rcm.o  ${PETSC_KSP_LIB}
	${RM} $(MAIN0).o


MAIN1 = main_1
$(MAIN1): $(MAIN1).o  chkopts
	-${CLINKER} -o $(MAIN1) $(MAIN1).o  ${PETSC_KSP_LIB}
	${RM} $(MAIN1).o

