function [ img ] = CTbackprojection( proj, param )
%CTBACKPROJECTION Summary of this function goes here
%   Detailed explanation goes here

img = zeros(param.nx, param.ny, param.nz, 'single');
textprogressbar('CTbackprojection: ')                   % Added   by Thilo
for i = 1:param.nProj
    textprogressbar((i-1)/(param.nProj-1)*100)
    img = img + backprojection(proj(:,:,i),param,i);
end
textprogressbar(' Done. ')                              % Added   by Thilo



end

