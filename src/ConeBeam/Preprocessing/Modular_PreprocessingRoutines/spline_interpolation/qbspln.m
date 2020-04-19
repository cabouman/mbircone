function y=qbspln(x)
% Calculate function values of a quadratic B-spline at point x
%
% See also: cbspln.m lbspln.m
%
% Usage: y=qbspln(x)

x=abs(x) ;
if x>1.5,
  y=0 ;
else
  if x>0.5,
    y=(3/2-x)^2/2 ;
  else
    y=0.75-x^2 ;
  end ;
end ;


