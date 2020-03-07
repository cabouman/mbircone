%CONVNSEP   Separable N-dimensional convolution.
%   C = CONVNSEP(h,V) performs an N-dimensional convolution or matrix V
%   with N kernels defined in the cell array h.
%   C = CONVNSEP(h,V,'shape') controls the size of the output. (See help
%   for Matlab's CONVN function).
%   Example: create a random 4-D array, convolve it with 4 kernels of
%   various lengths:
%   h={[1 -2 1], [-1 1 -1 1 -1],[1 -2 1],[1]};
%   V = convnsep(h,randn([32,32,32,32]));
%   See also CONVN, CONV, CONV2
%
%   Written by Igor Solovey, 2010
%   isolovey@uwaterloo.ca
%
%   Version 1.1, February 26, 2011
%   TODO: fix handling of even-sized kernels
function J = convnsep(h,V,type)
lh=length(h);

%input validation
assert(lh==ndims(V),'The number of kernels does not match the array dimensionality.');

L=nan(1,lh);
for j=1:lh,
    L(j)=(length(h{j})-1)/2;
end
V=padarray(V,L);
J = convnsepsame(h,V);

%implicit behaviour: if no 'type' input, then type=='full' (don't trim the
%result)
if nargin>2
    switch type
        case 'full'
            %do nothing
        case 'valid'
            J=trimarray(J,L*2);
        case 'same'
            J=trimarray(J,L);
        otherwise
    end
end

end

%Perform convolution while keeping the array size the same (i.e. discarding
%boundary samples)
function J = convnsepsame(h,V)
J=V;
sz=size(V);
n=length(sz);
indx2=nan(1,n);
for k=1:n
    %dimensions other k-th dimension, along which convolution will happen:
    otherdims = 1:n; otherdims(k)=[];
    
    % permute order: place k-th dimension as 1st, followed by all others:
    indx1=[k otherdims];
    
    % inverse permute order:
    indx2(indx1)=1:n;
    
    %perform convolution along k-th dimension:
    %
    %1. permute dimensions to place k-th dimension as 1st
    J = permute(J,indx1);
    %2. create a 2D array (i.e. "stack" all other dimensions, other than
    %k-th:
    J = reshape(J,sz(k),prod(sz(otherdims)));
    %3. perform 2D convolution with k-th kernel along the first dimension
    J = conv2(h{k},1,J,'same');
    %4. undo the "flattening" of step 2
    J = reshape(J,sz(indx1));
    %5. undo the permutation of step 1.
    J = permute(J,indx2);
end
end

%extract only the central portion of the array V, based on kernels whose
%lengths are defined in L
function V = trimarray(V,L)
str='';
for j=1:ndims(V)
    str=[str num2str(L(j)+1) ':' num2str(size(V,j)-L(j)) ','];
end
str=str(1:end-1); %remove last coma

eval(['V=V(' str ');']);
end
