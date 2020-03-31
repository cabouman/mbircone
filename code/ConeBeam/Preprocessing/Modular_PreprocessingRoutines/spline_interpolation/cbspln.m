function y=cbspln(x)
% Calculate the value of a cubic B-spline at point x
%
% Usage: y=cbspln(x) ;

x=abs(x) ;
if x>2,
  y=0 ;
else
  if x>1,
    y=(2-x)^3/6 ;
  else
    y=2/3-x^2*(1-x/2) ;
  end ;
end ;


