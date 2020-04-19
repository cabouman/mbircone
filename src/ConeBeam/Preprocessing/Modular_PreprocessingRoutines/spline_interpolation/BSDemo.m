% This demonstration script shows how to use some of the routines
% from the Matlab B-spline Repository

disp('First, let us show how B-Splines look') ;

dt=0.01 ;

t=-3:dt:3 ;
lt=length(t) ;
for i=1:lt,
  f3(i)=cbspln(t(i)) ;
  f2(i)=qbspln(t(i)) ;
  f1(i)=lbspln(t(i)) ;
end ;

figure(1) ; 
plot(t,[f1 ; f2 ; f3]) ;
title('B-Splines') ; legend('Linear','Quadratic','Cubic') ;
grid on ; zoom on ; 

disp('Press any key to go on') ;
pause ;

disp('We shall show the interpolation using B-splines')
disp('taking sin(x)/sqrt(x) and sampling it at z=1,2,3,4,5')
disp('finding B-spline interpolation coefficients and interpolating') ;

figure(2) ;

x=1:1:5 ;          % Range of sampling points
c=sin(x)./sqrt(x) ; % Our function

t=1:dt:5 ;         % Output range
lt=length(t) ;
c2=qbanal(c) ;
c3=cbanal(c) ;
for i=1:lt,
  y1(i)=lbinterp(c,t(i)) ;
  y2(i)=qbinterp(c2,t(i)) ;
  y3(i)=cbinterp(c3,t(i)) ;  
end ;  

clf ;
plot(t,[y1 ; y2 ; y3 ; sin(t)./sqrt(t) ]) ; hold on
plot(x,c,'c*') ;
grid on ;
title('Spline interpolation') ; 
legend('Linear','Quadratic','Cubic','Original function') ;
zoom on ;

figure(3) ;
plot(t,[y1-sin(t) ; y2-sin(t)./sqrt(t) ; y3-sin(t)./sqrt(t)]) ;
grid on ; zoom on ;
title('Interpolation errors') ; 
legend('Linear','Quadratic','Cubic') ;

disp('End of the demonstration.') ;


