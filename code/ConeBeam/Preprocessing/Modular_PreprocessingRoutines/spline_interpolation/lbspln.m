function y=lbspln(x)
% Calculate function values of a linear B-spline at point x
%
% Usage: y=lbspln(x) 

x=abs(x) ;
if x>1,
  y=0 ;
else
  y=1-x ;
end ;


