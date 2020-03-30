function [vol_new, coeff] = LS_fit_vol(vol, vol_ref, mask)

if ~exist('mask') 
	mask = ones(size(vol));
end
if isempty(mask)
	mask = ones(size(vol));
end

A = [vol(:) ones(length(vol(:)),1)];
b = vol_ref(:);
coeff = solveWeightedLS(A, b, mask(:));

%%
vol_new = coeff(1)*vol + coeff(2);

return
