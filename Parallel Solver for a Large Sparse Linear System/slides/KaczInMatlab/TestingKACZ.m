%% Generate a banded matrix. Make sure bandwidth is an odd integer
n = 67;                 % number of rows
bandwidth = 9;          % bandwidth, odd integer so upper and lower bandwidth is same
p = 4;                  % number of "processors" to use when reordering

A = FormBanded(n,bandwidth);    %form a matrix
f = A*ones(n,1);                %forming a right hand side

%% Correct Answer:
% Should be all ones:
x = A\f

%% Reorder and then test

[Amats,Alast,newf,B] = ReorderBanded(A,f,p);

spy(B)

xReordered = B'\newf

%% Test my Kacz algorithm
xRP = KACZ(A,f,p)

% should be close to zero for error:
norm(ones(n,1) - xRP)

%% Try our small test case:

load lns_131.mat

A = full(Problem.A);

% Make banded
pp = symrcm(A);
A = A(pp,pp);
spy(A)
n = size(A,1);

p = 10;

%% Form right hand side and reorder

f = A*ones(n,1);

[Amats,Alast,newf,B] = ReorderBanded(A,f,p);

spy(B')

f = newf;
%% Solve Using row projection method
xRP = KACZ(A,f,p)

norm(ones(n,1) - xRP)
