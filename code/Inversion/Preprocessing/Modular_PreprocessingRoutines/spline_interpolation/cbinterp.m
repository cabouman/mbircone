function y=cbinterp(c,x)
% Given coefficients c of cubic B-splines at points 1,2,... 
% (obtained, for example, from cbanal.m) calculate a value at point x
%
% See also: cbanal.m
% Uses: cbspln.m
%
% Usage: y=cbinterp(c,x) ;

lenc=length(c) ;
xf=floor(x)-1 ; 

if xf>=1 & xf<=lenc,
  y=c(xf)*cbspln(x-xf) ;
else 
  y=0 ;
end ;

if xf+1>=1 & xf+1<=lenc,
  y=y+c(xf+1)*cbspln(x-xf-1) ;
end ; 

if xf+2>=1 & xf+2<=lenc,
  y=y+c(xf+2)*cbspln(x-xf-2) ;
end ; 

if xf+3>=1 & xf+3<=lenc,
  y=y+c(xf+3)*cbspln(x-xf-3) ;
end ; 



