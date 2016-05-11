function [Amats,Alast,newf,B] = ReorderBanded(A,f,p)
%
% Permute the banded matrix A, and permute f correspondingly.
%   (We are setting ourselves up to solve Ax = f)
%
% Amat and Alast together define an array of smaller matrices that we will
% be using in the sequence of least squares function:
%       (I -P1)...(I-Pm)...(I-P1)u = v
%
%   p is the number of processors we are interested in using

n = size(A,1);
b = ceil(n/p^2);

nbyp = ceil(n/p);       % number of rows per A[j] when resorted

npp = ceil(nbyp/p);     % number of rows per process in each A[j]
b
npp

nbyptimesp = npp*p;

nbyp;

% Processors 1 to p-1 own p*npp rows total
% Processor p owns the rest of the n - (p-1)*p*npp

% We construct two vectors, Block and Proc
% Block(i) tells me which block old row i goes to
% Proc(i) tells me which process old row i goes to

% The first p*npp rows go to processor 1
% The next p*npp rows to to processor 2, etc
% ... the last n - (p-1)*p*npp rows go to processor p

Apsize = n - (p-1)*p*npp;


Proc = zeros(n,1);
ProcStart = zeros(p,1);

for i = 1:p
    ProcStart = (i-1)*p*npp;
    Proc(ProcStart+1:ProcStart+p*npp) = i;
end

Proc = Proc(1:n);

Block = zeros(n,1);

% Now we want to find out which block each row belongs to.
% The pattern goes:
%   - first npp go to block 1
%   - next npp go to block 2
%   - etc.
%   - the last nbyp - (p-1)*npp go to block p

% Make that pattern once:
next = 1;

for i = 1:p-1       % go through the chunk belonging to process i < p  
    for j = 1:p     % each block will get npp rows
        for k = 1:npp
            Block(next) = j;
            next = next+1;
        end
    end
end

for i = p
    for j = 1:p
        for k = 1:(nbyp - (p-1)*npp)
            Block(next) = j;
            next = next+1;
        end
    end
end

Block = Block(1:n);

id = [Block Proc (1:n)'];

id = sortrows(id,[1,2])

order = id(:,3);
B = A(order,:)';
newf = f(order);

% b is the number of columns belonging to each project in A_j
% p is the number of prcesses
% so b*p is the number of columns in the first p-1 A_j matrices


Bstart = zeros(p,1);

for i = 1:p
    Bstart(i) = 1 + (i-1)*nbyp;
end

Amats = B(:,1:nbyp);

for i = 2:p-1
    Amats(:,:,i) = B(:,Bstart(i):Bstart(i)+nbyp-1);
end

Alast = B(:,Bstart(p):n);

% Now I have the A_j pieces, and we can do whatever we want with them.


%     
% 
% Amats = zeros(n,nbyp);
% 
% 
% Block = zeros(n,1);
% next = 1;
% 
% start = zeros(p,1);
% count = ones(p,1);  %counter for each j = 1:p
% 
% for i = 1:p     % this is the future starting index of the block A[i]
%     start(i) = (i-1)*nbyp;
% end
% 
% % 'start' is the index in the final output where Aj starts
% 
% for i = 1:p         % iterate through the future A[i] blocks
%     for j = 1:p-1   % iterate through the p processors that each have a piece of A[i]
%         
%         % i refers to the future block A[i],
%         % each of which belongs to p processors.
%         % The first p-1 processors have npp rows, and then the last
%         % has the remainder nbyp-(p-1)*npp (= nbyp if perfect partition)
%         
%         % start(i) refers to the place where Ai starts
%         % counter(j) refers to the index within Ai
%         
%         for k = 1:npp
%             %if start(j) + count(j) < n+1
%                 Block(next) = start(j) + count(j);
%                 count(j) = count(j) + 1;
%                 next = next + 1;
%             %end
%         end
%     end
% end
% 
% size(Block)
% size(A)
% Block
% 
% 
% Block = Block(1:n);
% 
% 
% B = A(Block,:)';
% 
% % b is the number of columns belonging to each project in A_j
% % p is the number of prcesses
% % so b*p is the number of columns in the first p-1 A_j matrices
% 
% Bstart = zeros(p,1);
% 
% for i = 1:p
%     Bstart(i) = 1 + (i-1)*b*p;
% end
% 
% Amats = B(:,1:b*p);
% 
% size(Amats)
% Bstart
% 
% for i = 2:p-1
%     Amats(:,:,i) = B(:,Bstart(i):Bstart(i)+nbyp-1);
% end
% 
% Alast = B(:,Bstart(p):n);
% 
% % Now I have the A_j pieces, and we can do whatever we want with them.
% 
% newf = f(Block);

end