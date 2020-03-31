function y=cbderiv(x)
% Calculate the first derivative of a cubic B-spline at point x
% 
% Usage: y=cbderiv(x) 

xs=sign(x) ; x=abs(x) ;

if x>2,
  y=0 ;
else
  if x>1,
    y=-0.5*(2-x)^2 ;
  else
    y=(1.5*x-2)*x ;
  end ;
end ;

y=y*xs ;

