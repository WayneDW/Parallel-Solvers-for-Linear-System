
if [ $# -eq 1 ] 
then
    cores=$1
elif [ $# -eq 0 ]
then
    cores=4
fi

echo "The nodes available: " $cores

#---- Data Sets ------# ------ Size -----------#
dat1=lns_131          # 131 * 131              #
dat2=std1_Jac2_db     # 21,982 * 21,982        #
dat3=bayer01
dat4=venkat25         # 62,424 * 62,424        #
dat5=stomach          # 213,360 * 213,360      #
dat6=atmosmodd        # 1,270,432 * 1,270,432  #
#---------------------#------------------------#


# --> You can change here <---------------------------------------
dat=${dat2}
source ~/mc.defs
mpdboot -f mpd.hosts -n 4

make main
mpiexec -n ${cores} ./main ./data/${dat}.mtx ./data/${dat}.permutation_vec

rm main


#-log_summary
