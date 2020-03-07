function Ibf=bilateral4i(I, sigmaSx, sigmaSy, sigmaSz, sigmaSt, sigmaR, samSx, samSy, samSz, samSt, samR)
I=double(I);
[N M P Q]=size(I);

%find the number of bins in each spatial dimension
Xbins=ceil(N/samSx);
Ybins=ceil(M/samSy);
Zbins=ceil(P/samSz);
Tbins=ceil(Q/samSt);

disp('Num Bins:');
disp([Xbins Ybins Zbins Tbins]);


Np=Xbins*samSx;
Mp=Ybins*samSy;
Pp=Zbins*samSz;
Tt=Tbins*samSt;

% Np
% Mp
% Pp
% Tt

Xrem=ceil((Np-N)/2);
Yrem=ceil((Mp-M)/2);
Zrem = ceil((Pp-P)/2);
Trem = ceil((Tt-Q)/2);

%zero-pad the image as symmetrically as possible to
%attain a size divisible by the subsample rate
Ip = padarray(I,[Xrem Yrem Zrem Trem ],0);
I = Ip(1:Np,1:Mp,1:Pp,1:Tt);

clear Ip

%%%%%
% 1. subsampling in spatial domain
%
% quantize intensity values
% a) map the image I onto a range [0,1].
% b) map the image I onto a range [0,1/sR].
% c) round the values to obtain an image Iq in which each pixel value
% Iq(x0) represents the range bin to which x0 belongs.
Iq = round((I-min(I(:)))/range(I(:))/samR);

%what bin does each pixel belong to, based on its x,y,z coordinate?
[X Y Z T]=ndgrid(1:Np,1:Mp,1:Pp,1:Tt);
binX=floor((X-1)/samSx)+1;
binY=floor((Y-1)/samSy)+1;
binZ=floor((Z-1)/samSz)+1;
binT=floor((T-1)/samSt)+1;
clear X Y Z T
%find which range bin each location in the x-y-z-range space belongs to
binW=repmat(reshape(1:samSx*samSy*samSz*samSt,[samSx samSy samSz samSt]),[Xbins Ybins Zbins Tbins]);

D=zeros(Xbins,Ybins,Zbins,Tbins,samSx*samSy*samSz*samSt);
W=D;
D(sub2ind(size(D),binX(:),binY(:),binZ(:),binT(:),binW(:)))=I(:);
W(sub2ind(size(D),binX(:),binY(:),binZ(:),binT(:),binW(:)))=Iq(:);

clear Iq binX binY binZ binT binW

%- W represents which 4D bin each data point belongs to
%- D represents what the actual value of each data point is*
%both contain values oriented according to coordinate system:
%(subsampled-x,subsampled-y,subsampled-z,location-in-bin)
%- the ordering of values along the 4th dimension no longer matters after
%this point.

%* (D is simply the input image I suitably rearranged. numel(D)=numel(I).)
%if I was a 2D image of size 12 by 12, and bin size in X and Y was 3 pixels
%each,constructing D would amount to breaking the image into 3-by-3 chunks,
%then taking the 9 pixels in each chunk and arranging them in a vector,
%placing these vectors next to each other to create a 4-by-4-by-9
%rearrangement of I.

GData       =zeros(Xbins,Ybins,Zbins,Tbins,1/samR);
GWeights    =zeros(Xbins,Ybins,Zbins,Tbins,1/samR);

%for each range bin:
for k=1:(1/samR)
    %find the indices of pixels belonging to this range bin
    tmp  = (W==k);
    %make a copy of D but zero out all values except those belonging to
    %k-th bin
    tmp2 = zeros(size(D));
    tmp2(tmp)=D(tmp);
    %GWeights will contain the number of pixels in k-th range bin for each
    %x-y-z bin
    GWeights(:,:,:,:,k) = sum(tmp,5);
    %And GData will contain the sum of their values
    GData(:,:,:,:,k)    = sum(tmp2,5);
end

%2. Smoothing with gaussians
hSx=gkernel(sigmaSx/samSx);
hSy=gkernel(sigmaSy/samSy);
hSz=gkernel(sigmaSz/samSz);
hSt=gkernel(sigmaSt/samSt);
hR=gkernel(sigmaR/samR);

GWeights=convnsep({hSx,hSy,hSz,hSt,hR},GWeights,'same');
GData=convnsep({hSx,hSy,hSz,hSt,hR},GData,'same');



%calculate coordinates in the non-subsampled X-Y-Z-Range plain
%to which filtered bin values belong
[n m p q k]=size(GData);
Xloc=repmat(linspace(1,n,Np)',[1 Mp Pp Tt]);
Yloc=repmat(linspace(1,m,Mp), [Np 1 Pp Tt]);
Zloc=repmat(shiftdim(linspace(1,p,Pp),-1),   [Np Mp 1 Tt]);
Tloc=repmat(shiftdim(linspace(1,q,Tt),-2),   [Np Mp Pp 1]);
Rloc=(I-min(I(:)))/range(I(:))*(1/samR-1)+1;

%compute the pixel values that will go to locations computed above
Indx=GWeights~=0;
Data=zeros(size(GData));
Data(Indx)=GData(Indx)./GWeights(Indx);

clear GData GWeights Indx

%At this point, values at evenly spaced locations Xloc,Yloc,Zloc are 
%known (stored in Data), but for the rest of the image they are unknown.
%Interpolation will find them.

% size(Data)
% size(Xloc)
% size(Yloc)
% size(Zloc)
% size(Tloc)
% size(Rloc)

% resize of unitary dimension
if N==1
    Data = shiftdim(Data);
    Xloc = shiftdim(Xloc);
    Yloc = shiftdim(Yloc);
    Zloc = shiftdim(Zloc);
    Tloc = shiftdim(Tloc);
    Rloc = shiftdim(Rloc);
    Ibf=interpn(Data,Yloc,Zloc,Tloc,Rloc);
    Ibf = reshape( Ibf, [Np Mp Pp Tt]);
else
    Ibf=interpn(Data,Xloc,Yloc,Zloc,Tloc,Rloc);
end

clear Data Xloc Yloc Zloc Tloc Rloc

%remove zero-padding
Ibf=Ibf(Xrem+1:N+Xrem,Yrem+1:M+Yrem,Zrem+1:P+Zrem,Trem+1:Q+Trem);



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
