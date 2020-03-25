function y=qbinterp(c,x)
% Given coefficients c of quadratic B-splines at points 1,2,... 
% (obtained, for example, from qbanal.m) calculate a value at point x
%
% See also: qbanal.m cbinterp.m lbinterp.m
%
% Usage: y=qbinterp(c,x) ;


lenc=length(c) ;
xf=floor(x-0.5) ; 

if xf>=1 & xf<=lenc,
  y=c(xf)*qbspln(x-xf) ;
else 
  y=0 ;
end ;

if xf+1>=1 & xf+1<=lenc,
  y=y+c(xf+1)*qbspln(x-xf-1) ;
end ; 

if xf+2>=1 & xf+2<=lenc,
  y=y+c(xf+2)*qbspln(x-xf-2) ;
end ; 



