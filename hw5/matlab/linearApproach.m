function [] = linearApproach(p, q, P, Q, W, s)
% Define symbols for p q P Q a b tp tq 
syms ps qs Ps Qs a b tp tq;

% Define nonlinear transformation model
equP = a * Ps + b * Qs + tp - ps;
equQ = -b * Ps + a * Qs + tq - qs;

%  Compute coefficient matrix and f matrix
B = [];
for u = [a b tp tq]
    B = [B arrayfun(@(i) double(diff(subs(...
                                          equP,...
                                          {ps, Ps, Qs},...
                                          {p(i), P(i), Q(i)}), u)), 1:size(p, 2))];
    B = [B arrayfun(@(i) double(diff(subs(...
                                          equQ,...
                                          {qs, Ps, Qs},...
                                          {q(i), P(i), Q(i)}), u)), 1:size(p, 2))];
end
B = reshape(B, [2 * size(p, 2), size(B, 2) / (2 * size(p, 2))]);

f = (arrayfun(@(i) double(-subs(...
                                equP,...
                                {ps, Ps, Qs, a, b, tp, tq},...
                                {p(i), P(i), Q(i), 0, 0, 0, 0})), 1:size(p, 2)));

f = [f (arrayfun(@(i) double(-subs(...
                                   equQ,...
                                   {qs, Ps, Qs, a, b, tp, tq},...
                                   {q(i), P(i), Q(i), 0, 0, 0, 0})), 1:size(p, 2)))]';

N = B' * W * B;                      % Compute normal matrix
t = B' * W * f;                      % Compute t matrix
X = inv(N) * t;                      % Compute the unknown parameters
V = f - B * X;                       % Compute residual vector
res = (V' * W * V);              % Compute residual square

% Output results
fprintf('a: \t%.18f\nb: \t%.18f\ntp: \t%.18f\ntq: \t%.18f\n', X)
fprintf('V.T * P * V = \t\t%.18f\n', res);

% Compute error of unit weight
s0 = sqrt(res / (size(B, 1) - size(B, 2)));
fprintf('Error of unit weight : %.4f\n', s0);
