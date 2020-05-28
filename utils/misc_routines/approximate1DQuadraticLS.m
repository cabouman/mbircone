function [ b, mu, c] = approximate1DQuadraticLS( f, X, W)
% f  is a function to approximate
% f_tilde = b/2  (x-mu)^2 + c_tilde


%(1) minimize || f - P gam ||^2_W
% P = powers of the x's
% Then f_tilde = b/2 x^2 + dx + e

P = [ X.^2, X.^1, X.^0 ];

gam = solveWeightedLS(P, f, W);

b = [2*gam(1)];
d = [gam(2)];
e = gam(3);

mu = - d / b;
c = e - b/2 * mu^2;


end

