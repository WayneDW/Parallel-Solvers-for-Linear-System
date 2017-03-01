function v = LeastSquares(Ai,u)

% This is a function that solves the least squares problem:
% Ai*w = u, and then returns the residual: v = Ai*w - u
%
% Here Ai is a tall and skinny full rank matrix.
%
% This corresponds to the matrix multiplication:
%       (I - Pi)u = v,
%
% Where Pi = Ai*(Ai'*Ai)^{-1}Ai'
%
%
%


w = Ai\u;
v = u-Ai*w;

end