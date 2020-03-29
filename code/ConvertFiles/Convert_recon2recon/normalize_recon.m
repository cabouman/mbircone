function [x_new] = normalize_recon( x, opts )

if opts.darkbright_from_GUI == 1
	if ~isfield(opts,'sliceID')
		opts.sliceID = floor(size(x,3)/2);
	end
	slice = x(:,:,opts.sliceID);
	[opts.dark, opts.bright] = get_dark_bright(slice)
end
x = x - opts.dark;
x = x / (opts.bright-opts.dark);

x_new = x;

return

