function [x] = solveWeightedLS(A, b, w)

%               ||          ||2
%% arg min_x    || Ax - b   ||
%               ||          ||W
% W = diag(w)
% x = (A' W A)^-1  A' W b

ATW = A' .* repmat(w', size(A', 1), 1);
x = (ATW * A) \ (ATW * b);
%(ATW * A)
%ATW * b

end