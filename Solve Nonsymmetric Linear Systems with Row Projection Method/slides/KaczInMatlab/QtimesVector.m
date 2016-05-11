function y = QtimesVector(Amats,Alast,x,p)

% The sequential least squares problem:
%
% y = (I - P1)(I-P2)...(I-P_p)(I-P_{p-1})...(I - P2) x

y = x;

% change this later by getting rid of first iteration
for j = 2:p-1   
    y = LeastSquares(Amats(:,:,j),y);
end

% for j = p
y = LeastSquares(Alast,y);


for j = p-1:-1:1
    y = LeastSquares(Amats(:,:,j),y);
end



end