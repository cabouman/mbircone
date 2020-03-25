function [outArr] = polynomial_gam(inArr, gam)
	% computes outArr = gam_0 + gam_1 inArr + gam_2 inArr.^2 + ...


exponent = 0:length(gam)-1;
outArr = zeros(size(inArr));
for i = 1:length(gam)

	coeff = gam(i);
	if(exponent(i)==0)
		outArr = outArr + gam(i);
	else
		outArr = outArr + gam(i) * inArr.^exponent(i);
	end

end


end