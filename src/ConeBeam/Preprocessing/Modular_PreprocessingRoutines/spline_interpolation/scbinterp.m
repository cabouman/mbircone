function y=scbinterp(g,a,b,x)
% Given coefficients g for cubic B-splines regularly placed between a and b, 
% calculate value at x
% i.e. g=[f(a) f(a+h) f(a+2*h) ... f(b-h) f(h)]
%      a<=x<=b
%      y=f(x)
% It is a scaled version of cubic B spline interpolation
%
% Uses: cbinterp.m
%
% Usage: y=scbinterp(g,a,b,x)

y=cbinterp(g,(x-a)/(b-a)*(length(g)-1)+1) ;
