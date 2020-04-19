function c=cbanal(y)
% Given values (y) of a function in points 1,2,...n, 
% find coefficients c, such as for all k=1..n
%    y(k)=sum  c(i) cbanal(k-i),   for i=1..n
%
% i.e., find exact cubic B-spline interpolation 
% Forms and solves a linear equation set, which is exact but slow for large n
% 
% See also: fspline.m
%
% Usage: c=cbanal(y)


N=length(y) ;
A=zeros(N) ;
y=y(:) ;

if N==1, c=1.5*y ; else
  A(1,1:2)=[2/3 1/6] ;
  A(N,N-1:N)=[1/6 2/3] ;
  for i=2:N-1,
    A(i,i-1:i+1)=[1/6 2/3 1/6] ;
  end ;  

  c=A\y ;
  c=c(:)' ;
end ;  
