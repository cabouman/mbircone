function [val] = immse_my(x1, x2)

val = mean((x1(:)-x2(:)).^2);

return