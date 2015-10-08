clear;
clc;

% Define coefficient matrix
A = [1 0 0 1 0 -1 0;
     0 -1 0 1 1 0 0;
     0 1 0 0 0 0 1;
     1 1 1 0 0 0 0];

% Define sigma0 and weight matrix
s0 = 3.0;
w = [.5 .1 .1 .3 .2 .3 .1];
W = diag(s0./w)^2;

% Define f matrix
f = [.3 0 0 .1]';

% Define vector of obserbation
l = [1.2 2.4 -3.7 -.4 2.8 1.1 -2.4]';

% Compute Normal matrix
N = A * inv(W) * A';

% Compute K matrix
K = inv(N) * f;

% Compute residual vector
V = inv(W) * A' * K;

% Print results
disp('Residual : ');
disp(V);

% Compute corrected observation
L = l + V;
disp('Correct observation : ');
disp(L);

% Compute error of unit weight
[row, col] = size(A);
sigma0 = sqrt((V'* W * V) / row);
fprintf('Error of unit weight : %.4f\n', sigma0);
