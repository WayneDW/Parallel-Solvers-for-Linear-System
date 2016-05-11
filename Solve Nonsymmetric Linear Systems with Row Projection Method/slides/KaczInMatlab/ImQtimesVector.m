function y = ImQtimesVector(Amats,Alast,x,p)

%(I - Q)x = x - Qx

y = x - QtimesVector(Amats,Alast,x,p);

end