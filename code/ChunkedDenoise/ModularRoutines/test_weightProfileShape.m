N = 100;
w11 = 1;
w12 = round(2*N/3);
w21 = round(N/3);
w22 = N;
s = 0.5;


w1 = zeros(N,1);
w1(w11:w12) = triangleND(w12 - w11 +1);
w1 = w1 .^ s;

w2 = zeros(N,1);
w2(w21:w22) = triangleND(w22 - w21 +1);
w2 = w2 .^ s;

w = w1 + w2;


w1_ = w1 ./ w;
w2_ = w2 ./ w;

figure(1)
stem(w1_)
hold on
stem(w2_)
hold off
title(['weight profiles $\tilde{w_i}$ when s = ', num2str(s)], 'Interpreter', 'Latex');