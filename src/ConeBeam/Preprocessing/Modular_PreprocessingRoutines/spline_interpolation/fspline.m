function y=fspline(x,iorder)
% FSPLINE(X,N) returns the B-spline coefficients of order N of the signal x.
%
% Given values (x) of a function in points 1,2,...n, 
% finds coefficients c, such as for all k=1..n
%    x(k)=sum  c(i) spline_of_order_N(k-i),   for i=1..n
%
% i.e., finds B-spline interpolation 
%
% This function uses a FIR filter thus it is faster than qbanal and cbanal,
% with usually negligible loss in accuracy. The FIR kernel is so far
% implemented only in C as a MEX file.
%
% References : 
% M. Unser, A. Aldroubi and M. Eden, "Fast B-spline transforms for
% continuous image representation and interpolation", IEEE Trans. Pattern Anal.
% Machine Intell., vol. 13, pp. 277-285, March 1991.
% M. Unser, A. Aldroubi and M. Eden, "B-spline signal processing. Part II :
% efficient design and applications", IEEE Trans. Signal Processing, vol. 41,
% pp. 834-848, February 1993.
%
%
% See also: qbanal.m cbanal.m
% Uses: filiirs.c
%
% Usage: y=fspline(x,N)

 if iorder==0   
			y=x; return
  elseif iorder==1   
			y=x; return
  elseif iorder==2 
   z=-3+2.*sqrt(2);c0=8;
		elseif iorder==3
   z=-2+sqrt(3);c0=6;
		elseif iorder==4
   z=[-0.361341 -0.0137254];c0=384;
		elseif iorder==5
   z=[-0.430575 -0.0430963];c0=120;
		elseif iorder==6
   z=[-0.488295 -0.0816793 -0.00141415];c0=46080;
		elseif iorder==7
   z=[-0.53528 -0.122555 -0.00914869];c0=5040;
  else
     error('Order not available')
     return
  end
y=filiirs(x,z,c0);
