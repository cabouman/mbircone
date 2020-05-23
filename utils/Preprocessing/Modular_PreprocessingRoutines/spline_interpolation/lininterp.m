function y=lininterp(g,a,b,x)
% Given function samples g regularly placed between a and b, calculate
% value at x
% i.e. g=[f(a) f(a+h) f(a+2*h) ... f(b-h) f(h)]
%      a<=x<=b
%      y=f(x)
%
% Uses: lbinterp.m
%
% Usage: y=lininterp(g,a,b,x) 

y=lbinterp(g,(x-a)/(b-a)*(length(g)-1)+1) ;
