function wght_new = normalize_wght(wght, preprocessingParams)

N_beta = size(wght,3);

% normalize by number of views so regularization is invariant with respect to num. views
wght = wght / N_beta;

% normalize using number of view subsets. This makes regularization invariant to num. view subsets
% This is because [N_beta_all = N_beta * num_viewSubsets = const.] when varying num_viewSubsets.
wght = wght / preprocessingParams.num_viewSubsets;

% this makes weight scaler approximately invariant to downcaling
wght = wght / sqrt(preprocessingParams.downscale_dv*preprocessingParams.downscale_dw);

wght_new = wght;

return