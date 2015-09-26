function [x, y, z, MU, ST] = genPt(num)

%Generate random points
x = normrnd(-100, 0.2, 1, num);
y = normrnd(230, 0.3, 1, num);
z = normrnd(135, 0.1, 1, num);

%Compute mean and standard deviation values
mx = mean(x);
my = mean(y);
mz = mean(z);
stdx = std(x);
stdy = std(y);
stdz = std(z);
MU = [mx, my, mz];
ST = [stdx, stdy, stdz];
