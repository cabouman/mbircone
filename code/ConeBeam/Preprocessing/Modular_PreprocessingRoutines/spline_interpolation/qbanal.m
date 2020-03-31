function c=qbanal(y)
% Given values (y) of a function in points 1,2,...n, 
% find coefficients c, such as for all k=1..n
%    y(k)=sum  c(i) qbspln(k-i),   for i=1..n
%
% i.e., find exact quadratic B-spline interpolation 
% Forms and solves a linear equation set, which is exact but slow for large n
% 
% See also: fspline.m
%
% Usage: c=qbanal(y)


N=length(y) ;
A=zeros(N) ;
y=y(:) ;

if N==1, c=y/0.75 ; else 
  A(1,1:2)=[0.75 0.125] ;
  A(N,N-1:N)=[0.125 0.75] ;
  for i=2:N-1,
    A(i,i-1:i+1)=[0.125 0.75 0.125] ;
  end ;  
  c=A\y ;
  c=c(:)' ;
end ;
