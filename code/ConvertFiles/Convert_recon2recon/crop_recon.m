function [x_new] = crop_recon( x, opts )


x_new = x(opts.limits_lo(1):opts.limits_hi(1),opts.limits_lo(2):opts.limits_hi(2),opts.limits_lo(3):opts.limits_hi(3));


return