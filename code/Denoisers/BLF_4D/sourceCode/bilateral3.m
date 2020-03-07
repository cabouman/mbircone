%BILATERAL3   Fast bilateral filtering of 3D images.
%   This is an implementation of fast bilateral filtering for 3D images.
%   This filter smoothes the image while preserving edges, but in its most
%   straighforward implementation is very computationally demanding,
%   especially with large 3D images. Fast bilateral filtering (S. Paris and
%   F. Durand, MIT) is an approximation technique which drastically
%   improves the speed of computation. This implementation follows the
%   Matlab version of the 2D filter[1] with some modifications.
%   
%   sigmaSxy and sigmaSz - spatial smoothing parameters (standard
%   deviation of the Gaussian kernel)
%   sigmaR - the smoothing parameter in the "range" dimension
%   samS - the amount of downsampling performed by the fast
%   approximation in the spatial dimensions (x,y). In z direction, it is
%   derived from spatial sigmas and samS.
%   samR - the amount of downsampling in the "range" dimension.
%   verbose - optional, for debugging
%
%   [1] http://people.csail.mit.edu/jiawen/software/bilateralFilter.m
%
%   Written by Igor Solovey, 2011
%   isolovey@engmail.uwaterloo.ca
%   Acknowledgements: 2D implementations of Jiawen Chen, Oleg Michailovich
%   Version 1, March 21 2011
%   
%Notes:
% - the internal function, BILATERAL3I, allows you to vary all parameters,
%   if you want to.
% - spatial sigmas are assumed to be in the units of "pixel".
% - range sigma is assumed to be in the units of range values whose
% minimum is 0 and maximum is 255 (i.e. 8-bit).

function Ibf=bilateral3(I, sigmaSxy,sigmaSz,sigmaR,samS,samR,verbose)
if ~exist('verbose','var'), verbose=0; end
I=range1(double(I),255);
%simplification #1
%smoothing and subsampling in x and y spatial directions is equal
sigmaSx=sigmaSxy;
sigmaSy=sigmaSxy;
samSx=samS;
samSy=samS;

%simplification #2
% obtain samSz by assuming samSz/samS=sigmaSz/sigmaSxy
c=sigmaSz/sigmaSxy;
samSz=ceil(c*samS);

%optional simpilification #3:
% obtain samR by assuming samS/sigmaS = samR/sigmaR
if ~exist('samR','var')
    samR = sigmaR*samS/sigmaSxy;
end

%re-scale sigmaR, samR to normalize them ( [0, 255] --> [0, 1] )
% samR has to be such that 1/samR is an integer
%e.g. if supplied samR is 12 (divide range [0,255] into bins of size 12):
%
%normalize: samR=12/256=0.0469;
%1/samR=21.3333 ~=22 bins (round up).
%the new samR = 1/22 = 0.0455;
%later on, when samR is used, 1/samR will yield the number of range bins, 22.
sigmaR  = sigmaR/256;
samR    = 1/ceil(256/samR);

%for debugging purposes:
if verbose
    displ('sigmaSx',sigmaSx);
    displ('sigmaSy',sigmaSy);
    displ('sigmaSz',sigmaSz);
    displ('sigmaR',sigmaR);
    displ('samSx',samSx);
    displ('samSy',samSy);
    displ('samSz',samSz);
    displ('samR',samR);
    disp(['{' n2s(size(I,1)) ',' n2s(size(I,2)) ',' n2s(size(I,3)) ...
        ',256} --> {' n2s(ceil(size(I,1)/samSx)) ',' n2s(ceil(size(I,2)/samSy)) ...
        ',' n2s(ceil(size(I,3)/samSz)) ',' n2s(1/samR) '}']);
end

%run the filter
Ibf=bilateral3i(I,sigmaSx,sigmaSy,sigmaSz,sigmaR,samSx,samSy,samSz,samR);





%Auxiliary functions:

%displaying variable value
function displ(name,val)
disp([name '=' n2s(val) ';']);

%shorthand for num2str
function S = n2s(N,f)
if N~=fix(N)
    if ~exist('f','var'),S=num2str(N,'%4.2f');
    elseif f==0
        S=num2str(N);
    else
        S=num2str(N,f);
    end
else
    S=num2str(N);
end

%make a gaussian kernel
function [h,L] = gkernel(sd)
L=ceil(3.5*sd);
h=exp(-0.5*((-L:L)'/sd).^2);
h=h/sum(h);

function Y = range1(x,rg)
if ~exist('rg','var'),rg=1;end
Y=1/range(x(:))*(x-min(x(:)))*rg;

function y = range(x)
y=max(x)-min(x);