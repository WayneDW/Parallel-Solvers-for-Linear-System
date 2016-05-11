function x = KACZ(A,f,p)

% Given a banded matrix A, we want to solve Ax = f using row projection
% methods.
%
% First we will reorder both A and f and get the A_j pieces that are needed
% in the sequence of least Squares. Then we will use CG acceleration to
% solve the problem fully.
%

n = size(A,1);


[Amats,Alast,newf,B] = ReorderBanded(A,f,p); %this updates the right hand side
                                             %and the matrix A
            

% Form the new right hand side: c = T*f                                       
c = Getc(B',Amats,Alast,newf,p);


cTest = ImQtimesVector(Amats,Alast,ones(n,1),p);
%[c,cTest]
c = cTest;

xk = c;

% rk := Q*c
rk = QtimesVector(Amats,Alast,xk,p);


Nr0 = norm(rk);
pk = rk;
k = 1;

tol = 1e-14;
res = 1;

while res > tol && k < 3000
    
    ImQpk = ImQtimesVector(Amats,Alast,pk,p);   %IQpk := (I-Q)*pk
    
    denoma = pk'*ImQpk;
    numa = rk'*rk;
    
    alpha = numa/denoma;
    
    xk = xk + alpha*pk;
    rk = rk - alpha*ImQpk;
    
    beta = (rk'*rk)/numa;
    
    pk = rk + beta*pk;
   
    
    res = norm(rk)/Nr0;
    
    
    fprintf('at k = %d, residual = %f \n',k,res);
    k = k+1;
end

x = xk;

end