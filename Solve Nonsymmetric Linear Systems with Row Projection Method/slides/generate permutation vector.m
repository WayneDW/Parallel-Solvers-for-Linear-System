%http://yifanhu.net/GALLERY/GRAPHS/search.html
%lns_131, size:131*131;
%std1_Jac2_db, 21,982*21982;
%bayer01, size: 57,735*57735;
%venkat25, size: 62,424*62,424;
%stomach, size: 213,360*213,360;
%atmosmodd, size: 1,270,432 * 1,270,432.  




name = 'std1_Jac2_db'
path = 'C:\Users\Wei\Documents\Courses\Parallelism Numerical Linear Algebra\CS515\matlab\'
filename = strcat(path,name,'.mat');
load(filename)
A=Problem.A;
p = symrcm(A);
R = A(p,p);
% change to dense
%B = full(R);
bandwidth(R)
%subplot(1,2,2),spy(A),title(name)
%subplot(1,2,2),spy(R),title(name)

% 2nd ordering

fout = fopen(strcat(path,name,'.permutation_vec'),'w');
fprintf(fout,'%d\n',p');
fclose(fout);