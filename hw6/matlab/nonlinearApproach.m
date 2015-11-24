function [res_mat dX_mat] = nonlinearApproach(p, q, P, Q, W, s)
% Define symbols for σ tp tq θ, Δσ, Δθ, Δtp, Δtq
syms Sigs tps tqs ts dSigs dts dtps dtqs;

% Define symbols for observation values
ps = sym('p%d', [1 size(p, 2)]);
qs = sym('q%d', [1 size(p, 2)]);
Ps = sym('P%d', [1 size(p, 2)]);
Qs = sym('Q%d', [1 size(p, 2)]);

% Define nonlinear transformation model
equP = (Sigs * Ps * cos(ts) + Sigs * Qs * sin(ts) + tps - ps);
equQ = (-Sigs * Ps * sin(ts) + Sigs * Qs * cos(ts) + tqs - qs);
F = [equP(:); equQ(:)];

% Define initial resolution values, space list and loop counter
res_old = 1000;
res_new = 10^-2;
lc = 1;

% Define space lists to record intermediate information
res_mat = [res_new];
dX_mat = [];

% Compute initial values of unknown parameters
index = [1:size(p, 2) 1];
Sigma = [];
theta = [];
for i = 1:size(index, 2) - 1
    Sigma = [Sigma hypot(p(index(i + 1)) - p(index(i)), q(index(i + 1)) - q(index(i)))...
    / hypot(P(index(i + 1)) - P(index(i)), Q(index(i + 1)) - Q(index(i)))];
    theta = [theta atan2(Q(index(i + 1)) - Q(index(i)), P(index(i + 1)) - P(index(i)))...
    - atan2(q(index(i + 1)) - q(index(i)), p(index(i + 1)) - p(index(i)))];
end

Sigma0 = mean(Sigma);  % Get mean value of scale factrs as initial parameter of Sigma
theta0 = mean(theta);  % Get mean rotate angle as initial parameter of theta

% Compute initial horizontal and vertical translation
% Get mean shift values as initial parameters
tp0 = mean((p - Sigma0 * (P * cos(theta0) + Q * sin(theta0))));
tq0 = mean((q - Sigma0 * (P * -sin(theta0) + Q * cos(theta0))));

% Use value of observation as its initial value
P0 = P;
Q0 = Q;
p0 = p;
q0 = q;
l0 = [p0(:) q0(:) P0(:) Q0(:)]';
l0 = l0(:);
l = [p(:) q(:) P(:) Q(:)]';              % Observations with matrix form
l = l(:);                                   % Observations with matrix form
X = ones(1);                                % Initial value for iteration

% Iteration process
while abs(sum(X)) > 10^-15
    % Linearize nonlinear model
    % Compute Jacobian matrix A and B, and F0 for constant term
    A = [];
    B = [];
    F0 = [];
    ls_ = [ps(:) qs(:) Ps(:) Qs(:)]';
    Xs = [Sigs ts tps tqs];
    for f = F
        % Compute Jacobian matrix of F function with respect to l
        for e = ls_(:)';
            A = [A double(diff(subs(F, {Sigs, ts, tps, tqs},...
            {Sigma0, theta0, tp0, tq0}), e))];
        end

        % Compute Jacobian matrix of F function with respect to x
        for x = Xs
            B = [B subs(diff(f, x), {Sigs, ts, tps, tqs},...
            {Sigma0, theta0, tp0, tq0})];
        end
        % Substitute symbols with initial values
        F0 = [F0 subs(f, {Sigs, ts, tps, tqs}, {Sigma0, theta0, tp0, tq0})];
    end
    % Substitute symbols with observation and initial values
    for i = 1:size(p, 2)
        for j = 1:size(B, 2)
            B(i, j) = double(subs(B(i, j), {Ps(i), Qs(i), ps(i), qs(i)},...
            {P0(i), Q0(i), p0(i), q0(i)}));
            B(i + size(p, 2), j) = double(subs(B(i + size(p, 2), j),...
            {Ps(i), Qs(i), ps(i), qs(i)}, {P0(i), Q0(i), p0(i), q0(i)}));
        end
        F0(i, 1) = subs(F0(i, 1), {Ps(i), Qs(i), ps(i), qs(i)},...
        {P0(i), Q0(i), p0(i), q0(i)});
        F0(i + size(p, 2), 1) = subs(F0(i + size(p, 2), 1),...
        {Ps(i), Qs(i), ps(i), qs(i)}, {P0(i), Q0(i), p0(i), q0(i)});
    end
    B = double(B);
    F0 = double(F0);
    f = -F0 - A * (l - l0);

    Qe = A * inv(W) * A';
    We = inv(Qe);
    N = (B' * We * B);                  % Compute normal matrix
    t = (B' * We * f);                  % Compute t matrix
    X = inv(N) * t;                     % Compute the unknown parameters
    V = inv(W) * A' * We * (f - B * X); % Compute residual vector

    % Update residual values
    res_old = res_new;
    res_new = (V' * W * V);
    res_mat = [res_mat res_new];

    % Update initial values
    Sigma0 = Sigma0 + X(1, 1);
    theta0 = theta0 + X(2, 1);
    tp0 = tp0 + X(3, 1);
    tq0 = tq0 + X(4, 1);
    l0 = (l + V);
    p0 = l0(1:4:size(l0, 1));
    q0 = l0(2:4:size(l0, 1));
    P0 = l0(3:4:size(l0, 1));
    Q0 = l0(4:4:size(l0, 1));
    dX_mat = [dX_mat X];

    % Output results
    fprintf([repmat('*', 1, 10) '  Iteration count: %d  ' repmat('*', 1, 10) '\n'], lc);
    fprintf('Δσ: \t%.18f\nΔθ: \t%.18f\nΔtp: \t%.18f\nΔtq: \t%.18f\n', X);
    fprintf('σ: \t%.18f\nθ: \t%.18f\ntp: \t%.18f\ntq: \t%.18f\n', Sigma0, theta0, tp0, tq0);
    fprintf('V.T * P * V = \t\t%.18f\n', res_new);
    fprintf('Δ(V.T * P * V) = \t%.18f\n', abs(res_new - res_old));
    fprintf([repmat('*', 1, 17) '  End  ' repmat('*', 1, 18) '\n']);

    % Update loop counter
    lc = lc + 1;
end

% Compute error of unit weight
s0 = sqrt(res_new / (size(B, 1) * 2 - size(B, 2)));
fprintf('Error of unit weight : %.4f\n', s0);

% Compute other informations
SigmaXX = s^2 * inv(N);
SigmaVV = s^2 * (inv(W) * A' - inv(W) * A' * We * B * inv(N) * B')...
    * (inv(W) * A' * We - inv(W) * A' * We * B * inv(N) * B' * We)';

Sigmallhat = (s^2 * inv(W)) - SigmaVV;

% Write out sigma matrix results
fout = fopen('SigmaMat.txt', 'w');
fprintf(fout, '∑dXdX = \n', 'n', 'utf-8');
for r = 1:size(SigmaXX, 1)
    for c = 1:size(SigmaXX, 2)
        fprintf(fout, '%.10f  ', SigmaXX(r, c));
    end
    fprintf(fout, '\n');
end
fprintf(fout, '\n');

fout = fopen('SigmaMat.txt', 'a');
fprintf(fout, '∑VV = \n', 'n', 'utf-8');
for r = 1:size(SigmaVV, 1)
    for c = 1:size(SigmaVV, 2)
        fprintf(fout, '%.10f  ', SigmaVV(r, c));
    end
    fprintf(fout, '\n');
end
fprintf(fout, '\n');

fout = fopen('SigmaMat.txt', 'a');
fprintf(fout, '∑llhat = \n', 'n', 'utf-8');
for r = 1:size(Sigmallhat, 1)
    for c = 1:size(Sigmallhat, 2)
        fprintf(fout, '%.10f  ', Sigmallhat(r, c));
    end
    fprintf(fout, '\n');
end
fprintf(fout, '\n');
fclose(fout);

fprintf('Covariance matrices have been written to file: ''SigmaMat.txt''...\n');
