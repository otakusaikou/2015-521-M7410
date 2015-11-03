clear;
clc;

% Disable scientific notation
format long g;

% Define input values
p = [16.6791, 47.6718, 72.4188, 8.4674, 15.7592, -24.3569];
q = [16.1734, 58.7223, 20.8377, 103.4796, -15.7964, 2.3997];
P = [10, 23, 60, -30, 21, -23];
Q = [20, 71, 45, 98, -10, -8];

% Define sigma0 and weight matrix
s = 0.01;       % Define a priori error
W = diag(s./0.01 * ones(1, 12))^2;

% Solve problem with nonlinear approach
fprintf('Solve problem with nonlinear approach...\n');

[res_mat, X_mat] = nonlinearApproach(p, q, P, Q, W, s);

% Draw delta X as functions of iteration number
dSigma = X_mat(1, :);
dtheta = X_mat(2, :);
dtp = X_mat(3, :);
dtq = X_mat(4, :);

% Create figure 1
figure('Name', 'fig1', 'position', [0 0 1000 1000]);

% Variation of scale
drawFunctionPlot(...
                 dSigma,...
                 {...
                     'Relationship between',...
                     'variation of scale and iteration times'},...
                 'Variation of scale',...
                 5,...
                 221,...
                 [0 6 -1 * 10^-5 9 * 10^-5],...
                 2 * 10^-5)

%% Variation of rotate angle
drawFunctionPlot(...
                 dtheta,...
                 {...
                     'Relationship between',...
                     'variation of rotate angle and iteration times'},...
                 'Variation of rotate angle (rad)',...
                 5,...
                 222,...
                 [0 6 -1 * 10^-5 9 * 10^-5],...
                 2 * 10^-5)

%% Variation of horizontal shift
drawFunctionPlot(...
                 dtp,...
                 {...
                     'Relationship between',...
                     'variation of horizontal shift and iteration times'},...
                 'Variation of horizontal shift (m)',...
                 4,...
                 223,...
                 [0 6 -3.5 * 10^-3 1.5 * 10^-3],...
                 10^-3);

%% Variation of vertical shift
drawFunctionPlot(...
                 dtq,...
                 {...
                     'Relationship between',...
                     'variation of vertical shift and iteration times'},...
                 'Variation of vertical shift (m)',...
                 4,...
                 224,...
                 [0 6 -3.5 * 10^-3 1.5 * 10^-3],...
                 10^-3);

% Draw delta residuals as functions of iteration number
% New residual values divided by the old one
div = res_mat(2:size(res_mat, 2)) ./ res_mat(1:size(res_mat, 2) - 1); 

% Create figure 2
figure('Name', 'fig2', 'position', [0 0 1000 1000]);

drawFunctionPlot(...
                 res_mat,...
                 {...
                     'Relationship between',...
                     'variation of residual and iteration times'},...
                 'Variation of residual',...
                 4,...
                 211,...
                 [-1, 6, 0, 0.012],...
                 0.002,...
                 -1);

drawFunctionPlot(...
                 div,...
                 {...
                     'Relationship between',...
                     'variation of division of residuals and iteration times'},...
                 'Variation of division of residuals',...
                 4,...
                 212,...
                 [0, 6, 0, 1.2],...
                 0.2);


