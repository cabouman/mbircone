function y=lbinterp(c,x)
% Given coefficients c of linear B-splines at points 1,2,... 
% (equal to functional values) calculate a value at point x
%
% See also: cbinterp.m qbinterp.m
% Uses: lbspln.m
%
% Usage: y=lbinterp(c,x) ;


lenc=length(c) ;
xf=floor(x) ; xc=xf+1 ;

if xf>=1 & xf<=lenc,
  y=c(xf)*lbspln(x-xf) ;
else 
  y=0 ;
end ;

if xc>=1 & xc<=lenc,
  y=y+c(xc)*lbspln(x-xc) ;
end ; 



