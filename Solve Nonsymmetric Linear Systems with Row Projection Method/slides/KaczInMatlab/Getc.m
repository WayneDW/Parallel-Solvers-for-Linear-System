function c = Getc(A,Amats,Alast,f,p)

n = size(Amats,1);
m = size(Amats,2);
l = size(Alast,2);

fprintf('m = %d \n',m);
fprintf('l = %d \n',l);


% Aj = Amats(:,:,j); up to p-1
% Ap = Alast;

D = zeros(n,n);

AAT = A*A';

start = zeros(p,1);

for i = 1:p
    start(i) = 1+(i-1)*m;
end

for i = 1:p-1
    D(start(i):start(i)+m-1,start(i):start(i)+m-1) = Amats(:,:,i)'*Amats(:,:,i);
end

D(start(p):n,start(p):n) = Alast'*Alast;

% So D is done;

LpLT = AAT - D;

LT = triu(LpLT);
L = LT';

DpL = D + L;


%c = T*f = (A'*inv(DpLT))*D*(inv(DpL)*f);
aa = DpL\f;

bb = D*aa;

DpLT = DpL';
cc = DpLT\bb;

c = A'*cc;




% Ignore below.

% fmost = f(1:(p-1)*m);
% 
% Fmost = reshape(fmost,m,p-1); % each column of Fmost is f_j corresponding to A_j
% 
% fleft = f((p-1)*m+1:n);   % this is f_l
% 
% 
% % now perform (A_i')^+ f_i
% 
% gleft = pinv(Alast')*fleft;
% 
% 
% Gmost = zeros(n,p-1);
% 
% for i = 1:p-1
%     Gmost(:,i) = pinv(Amats(:,:,i)')*Fmost(:,i);
% end
% 
% GmostOld = Gmost;
% 
% 
% % Calculate a piece of the summand of c, for piece 1
% i = 1;
% 
% for j = (p-1):-1:(i+1)
%     Gmost(:,1) = LeastSquares(Amats(:,:,j),Gmost(:,1));
% end
% 
% for j = i:p-1
%     Gmost(:,1) = LeastSquares(Amats(:,:,j),Gmost(:,1));
% end
% %j = p
% Gmost(:,1) = LeastSquares(Alast,Gmost(:,1));
% 
% 
% Gmost(:,1) = GmostOld(:,1) + Gmost(:,1);
% 
% % For piece i, 1 < i < p-1
% 
% for i = 2:p-2
%     
%     for j = p-1:-1:(i+1)
%         Gmost(:,i) = LeastSquares(Amats(:,:,j),Gmost(:,i));
%     end
% 
%     for j = i:p-1
%         Gmost(:,i) = LeastSquares(Amats(:,:,j),Gmost(:,i));
%     end
%     %j = p
%     Gmost(:,i) = LeastSquares(Alast,Gmost(:,i));
% 
%     Gmost(:,i) = GmostOld(:,i) + Gmost(:,i);
% 
%     for j = 1:i-1
%         Gmost(:,i) = LeastSquares(Amats(:,:,j),Gmost(:,i));
%     end
%     
% end
% 
% % i = m-1
% i = p-1;
% 
% for j = i:p-1
%     Gmost(:,i) = LeastSquares(Amats(:,:,j),Gmost(:,i));
% end
% 
% %j = p
% Gmost(:,i) = LeastSquares(Alast,Gmost(:,i));
% 
% Gmost(:,i) = GmostOld(:,i) + Gmost(:,i);
% 
% for j = 1:i-1
%     Gmost(:,i) = LeastSquares(Amats(:,:,j),Gmost(:,i));
% end
% 
% % i = p, where we deal with gleft:
% i = p;
% for j = 1:i-1
%     gleft = LeastSquares(Alast,gleft);
% end
% 
% 
% % c = gleft;
% % 
% % for i = 1:p-1
% %     c = c + Gmost(:,i);
% % end


end