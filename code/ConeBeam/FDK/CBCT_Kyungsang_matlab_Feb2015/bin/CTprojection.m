function [ proj ] = CTprojection( img, param )
%CTPROJECTION Summary of this function goes here
%   Detailed explanation goes here

proj = zeros(param.nu, param.nv, param.nProj,'single');

for i = 1:param.nProj    
    proj(:,:,i) = projection(img,param,i);
end
    
end

