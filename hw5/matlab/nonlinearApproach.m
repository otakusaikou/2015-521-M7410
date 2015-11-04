function [res_mat dX_mat] = nonlinearApproach(p, q, P, Q, W, s)
% Change encoding to 'utf-8'
% slCharacterEncoding('UTF-8');

% Define symbols for p q P Q σ tp tq θ, Δσ, Δθ, Δtp, Δtq
syms ps qs Ps Qs Sigs tps tqs ts dSigs dts dtps dtqs;

% Define nonlinear transformation model
equP = Sigs * Ps * cos(ts) + Sigs * Qs * sin(ts) + tps;
equQ = -Sigs * Ps * sin(ts) + Sigs * Qs * cos(ts) + tqs;

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

% Iteration process
while abs(res_new - res_old) > 10^-18
    % Linearize nonlinear model
    linP = subs(...
                (equP + diff(equP, Sigs) * dSigs + diff(equP, ts) * dts + diff(equP, tps)...
                    * dtps + diff(equP, tqs) * dtqs - ps),...
                {Sigs, ts, tps, tqs},...
                {Sigma0, theta0, tp0, tq0});

    linQ = subs(...
                (equQ + diff(equQ, Sigs) * dSigs + diff(equQ, ts) * dts + diff(equQ, tps)...
                    * dtps + diff(equQ, tqs) * dtqs - qs),...
                {Sigs, ts, tps, tqs},...
                {Sigma0, theta0, tp0, tq0});

    %  Compute coefficient matrix and f matrix
    B = [];
    for u = [dSigs dts dtps dtqs]
        B = [B arrayfun(@(i) double(diff(subs(...
                                              linP,...
                                              {ps, Ps, Qs},...
                                              {p(i), P(i), Q(i)}), u)), 1:size(p, 2))];
        B = [B arrayfun(@(i) double(diff(subs(...
                                              linQ,...
                                              {qs, Ps, Qs},...
                                              {q(i), P(i), Q(i)}), u)), 1:size(p, 2))];
    end
    B = reshape(B, [2 * size(p, 2), size(B, 2) / (2 * size(p, 2))]);

    f = (arrayfun(@(i) double(-subs(...
                                    linP,...
                                    {ps, Ps, Qs, dSigs, dts, dtps, dtqs},...
                                    {p(i), P(i), Q(i), 0, 0, 0, 0})), 1:size(p, 2)));

    f = [f (arrayfun(@(i) double(-subs(...
                                       linQ,...
                                       {qs, Ps, Qs, dSigs, dts, dtps, dtqs},...
                                       {q(i), P(i), Q(i), 0, 0, 0, 0})), 1:size(p, 2)))]';

    N = B' * W * B;                      % Compute normal matrix
    t = B' * W * f;                      % Compute t matrix
    X = inv(N) * t;                      % Compute the unknown parameters
    V = f - B * X;                       % Compute residual vector

    % Update residual values
    res_old = res_new;
    res_new = (V' * W * V);
    res_mat = [res_mat res_new];

    % Update initial values
    Sigma0 = Sigma0 + X(1, 1);
    theta0 = theta0 + X(2, 1);
    tp0 = tp0 + X(3, 1);
    tq0 = tq0 + X(4, 1);
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
s0 = sqrt(res_new / (size(B, 1) - size(B, 2)));
fprintf('Error of unit weight : %.4f\n', s0);

% Compute other informations
SigmaXX = s^2 * inv(N);
SigmaVV = s^2 * (inv(W) - B * inv(N) * B');
Sigmall = s^2 * inv(W);

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
fprintf(fout, '∑ll = \n', 'n', 'utf-8');
for r = 1:size(Sigmall, 1)
    for c = 1:size(Sigmall, 2)
        fprintf(fout, '%.4f  ', Sigmall(r, c));
    end
    fprintf(fout, '\n');
end
fprintf(fout, '\n');
fclose(fout);

fprintf('Covariance matrics have been written to file: ''SigmaMat.txt''...\n');
