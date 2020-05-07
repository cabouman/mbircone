function y=qbderiv(x)
% Calculate first derivative of a quadratic B-spline at point x
%
% Usage: y=qbderiv(x) 

xs=sign(x) ; x=abs(x) ;

if x>1.5,
  y=0 ;
else
  if x>0.5,
    y=-1.5+x ;
  else
    y=-2*x ;
  end ;
end ;

y=y*xs ;

