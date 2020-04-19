function [ B, mu, c] = approximate2DQuadraticLS( f, X1, X2, W)
% f_1 = f(X1_i, X2_i) is a function to approximate
% f_tilde = 1/2  (x-mu)' B (x-mu) + c_tilde


%(1) minimize || f - P gam ||^2_W
% P = powers of the x's
% Then f_tilde = 1/2 x' B x + d' x + e

P = [ X1.^2, X1.*X2, X2.^2, X1, X2, X1.^0 ];

gam = solveWeightedLS(P, f, W);

B = [2*gam(1) gam(2); gam(2) 2*gam(3)];
d = [gam(4); gam(5)];
e = gam(6);

mu = - inv(B) * d;
c = e - 1/2 *mu' * B * mu;


end

