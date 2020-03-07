% This script can plot the samples and the quadratic approximation that
% come from the tuning script

paramValues_list = [0.49547      0.5288     0.56214     0.59547      0.6288     0.66214     0.69547]; paramValues_list = paramValues_list'; error_list = [1.8999e-07   1.593e-07  1.4049e-07  1.3388e-07  1.3946e-07  1.5723e-07  1.8715e-07]; error_list = error_list'; 


f = error_list;
X = paramValues_list;
W = ones(size(f));

[ b, mu, c] = approximate1DQuadraticLS( f, X, W);


figure(4);
stem(X, f);

rad = max(abs(X-mu));
x_tilde = linspace(mu-rad, mu+rad, 100);
f_tilde = b/2 * (x_tilde-mu).^2 + c;

hold on
plot(x_tilde, f_tilde);
hold off

legend('Samples', 'Quadratic Approximation')
title(['Apptoximation with [b,mu,c] = [', num2str([b,mu,c]), ']'])
