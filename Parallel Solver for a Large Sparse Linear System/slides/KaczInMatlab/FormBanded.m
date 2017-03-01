function A = FormBanded(n,bandwidth)
% This generates a banded n by n matrix, with the given bandwidth.
% Used for testing the row projection method code

lu = (bandwidth-1)/2;
Aband = rand(bandwidth,n);

A = zeros(n+2*lu,n);

for i = 1:n
    A(i+lu,i) = Aband(lu+1,i);
    for k = 1:lu
       A(i+lu-k,i) = Aband(lu+1-k,i);
       A(i+lu+k,i) = Aband(lu+1+k,i);
    end
end

A = A(lu+1:n+lu,:);


end