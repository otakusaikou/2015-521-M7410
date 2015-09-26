clear;
%define D (y axis) and A (x axis)
x = [20:5:80];
y = [.02, .04, .07, .11, .2, .23, .32, .35, .46, .5, .62, .65, .79];

%A matrix
A = [transpose(x),ones(13, 1)];

%L matrix
L = transpose(y);

%Calculate X matrix
N = (transpose(A) * A);
W = (transpose(A) * L);
X = N\W;

%Calculate V matrix and sigma 0
V = A * X - L;
[n, t] = size(A);
sig0 = sqrt((transpose(V) * V) / (n - t));

%Calculate Q matrix, sigmaX and sigmaY
Q = inv(N);
sigX = sig0 * sqrt(Q(1, 1));
sigY = sig0 * sqrt(Q(2, 2));

%plot result
grid on;
hold on;
axis([0, 100, 0, 1]);

scatter(x, y, 'filled', 'r');
title('Relationship between factor A and D', 'FontSize', 15);
xlabel('Age',  'FontSize', 10);
ylabel('Disease Rate',  'FontSize', 10);

slope = X(1, 1);
intercept = X(2, 1);
x2 = [20:20:101];
plot(x2, x2 * slope + intercept);
plot(x2, x2 * slope + intercept + sig0, 'r--');
plot(x2, x2 * slope + intercept - sig0, 'r--');