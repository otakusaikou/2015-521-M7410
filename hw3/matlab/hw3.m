clear;
clc;

% Define coefficient matrix
B = [-1 0 0;
     0 0 1;
     1 0 -1;
     0 1 0;
     0 -1 1;
     -1 1 0;
     0 0 -1];

% Define sigma0 and weight matrix
s0 = 3.0;
s = [.5 .1 .1 .3 .2 .3 .1];
P = diag(s0./s)^2;

% Define f matrix
f = [103.8 -107.4 3.7 -104.6 -2.8 -1.1 107.4]';

% Define vector of obserbation
l = [1.2 2.4 -3.7 -.4 2.8 1.1 -2.4]';

% Compute Normal matrix
N = B' * P * B;

% Compute W matrix
W = -B' * P * f;

% Compute the unknown parameters
X = inv(N) * W;

% Compute residual vector
V = B * X + f;

% Print results
disp('Residual : ');
disp(V);

% Compute corrected observation
L = l + V;
disp('Correct observation : ');
disp(L);

% Compute error of unit weight
[row, col] = size(B);
sigma0 = sqrt((V'* P * V) / (row - col));
fprintf('Error of unit weight : %.4f\n', sigma0);
