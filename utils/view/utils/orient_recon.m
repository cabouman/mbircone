function [x_new] = orient_recon( x, opts )

% Permute
x = permute(x, opts.indexOrder);

%% Do flips
if(opts.flipVect(1) == 1)
	x = flip(x, 1);
end
if(opts.flipVect(2) == 1)
	x = flip(x, 2);
end
if(opts.flipVect(3) == 1)
	x = flip(x, 3);
end

x_new = x;

return

