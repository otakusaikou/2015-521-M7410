function [] = drawFunctionPlot(data, title_, ylabel_, ydigit, pos, xylim, offset, xshift)
% Define default value for variable xshift
if ~exist('xshift', 'var')
    xshift = 0;
end
subplot(pos);

% Use grid and allow MATLAB to draw new plot without erasing the old one
grid on;
hold on;

% Set title and xy labels
title(title_, 'FontSize', 15);
xlabel('Iteration time', 'FontSize', 15);
ylabel(ylabel_, 'FontSize', 15);

% Plot given data and shift x value if variable xshift is available
plot((1:size(data, 2)) + xshift, data, 'bo');
plot((1:size(data, 2)) + xshift, data, 'b-');

% Set y label interval and display format of numbers
set(gca,'YTick', xylim(3):offset:xylim(4));
set(gca,'YTickLabel', sprintf(['%.' int2str(ydigit) 'f|'], get(gca,'YTick')));
axis(xylim);    % Set range of x y axis
