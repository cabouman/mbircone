function [y_new, y_new_tilde, Ax_s_tilde, y_0_tilde, Yga, Hmu] = BHC_SC_atomic(Ax_s, y_0, par)

if(par.isEnergyDomain==1)
	Ax_s_tilde = exp(-Ax_s);
	y_0_tilde = exp(-y_0);
else
	Ax_s_tilde = Ax_s;
	y_0_tilde = y_0;
end


%% Powers of y_0_tilde
exp_range = par.exp_range;


len_ga = length(exp_range);
len_mu = 1 + length(par.sigma_H);
len_gamu = len_ga + len_mu;

indicesY = 1:len_ga;
indicesH = (1:len_mu) + len_ga;

YH = zeros(length(Ax_s_tilde(:)), len_gamu);


%% Prepare Y = powers of y_0_tilde
YH(:,indicesY) = repmat(abs(y_0_tilde(:)), 1, length(exp_range)) .^ repmat(exp_range, length(y_0_tilde(:)), 1);



%% Prepare H = blurred sino
for i_H = 1:length(indicesH)

    if(i_H==1)
        temp = ones(size(y_0));
    else
        temp = imgaussfilt(y_0.^par.power_H(i_H-1), par.sigma_H(i_H-1), 'padding', 'replicate');
    end
    i_YH = indicesH(i_H);

    YH(:,i_YH) = temp(:);

end
clear temp

for i = 1:length(size(YH,2))
    temp = YH(:,i);
    YH(:,i) = temp/mean(temp(:));
end

% w = const.
w = 1./y_0_tilde;
w = w/mean(w(:));
%w = ones(size(y_0_tilde));

%% ----------------- BHC -------------------------------------------------------------

% Solve LS || YH (ga,mu)' - (Ax_s_tilde)||^2_W for (ga,mu)
[gamu_hat] = solveWeightedLS(YH, Ax_s_tilde(:), w(:));


gamu = gamu_hat;

ga = gamu(indicesY);
mu = gamu(indicesH);

%disp(['ga = [', num2str(ga'), ']; % normalized'])
%disp(['mu = [', num2str(mu'), ']; % normalized'])

Yga = reshape(YH(:,indicesY)*ga,     size(y_0_tilde));
Hmu = reshape(YH(:,indicesH)*mu,     size(y_0_tilde));

y_new_tilde = Yga + Hmu;


if(par.isEnergyDomain==1)
	y_new = -log(max(y_new_tilde,exp(-8)));
else
	y_new = y_new_tilde;
end


end



