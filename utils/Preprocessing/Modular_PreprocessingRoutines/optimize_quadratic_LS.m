function [ minimizer, b,  x_pol, y_pol] = optimize_quadratic_LS(y, x, verbose)
% minimize || y - Ab ||^2
% y = f(x)       x = column vector: sample points of f

Nx = length(x);
A = zeros(Nx, 3);
A = [ x.^0 x.^1 x.^2 ];

b = inv(A' * A) * A' * y;

% minimize quadratic polynomial = b(1) + b(2)x + b(3)x^2
minimizer = - b(2) / (2*b(3));
disp(['Minimizer = ', num2str(minimizer)]);


% give fine sampling of LS fit
t1 = min( [x; minimizer] );
t2 = max( [x; minimizer] );

MIN = t1 - 0.1*(t2-t1);
MAX = t2 + 0.1*(t2-t1);

x_pol = linspace(MIN, MAX, 1000);
y_pol = b(1)*x_pol.^0 + b(2)*x_pol.^1 + b(3)*x_pol.^2;


if(verbose==true)
    scatter(x, y)
    hold on
    plot(x_pol, y_pol)
    hold off

    line([minimizer, minimizer], [min(y_pol), max(y_pol)]);
    legend('Samples', 'Estimated polynimial', ['Minimizer = ', num2str(minimizer)]);
end


end
