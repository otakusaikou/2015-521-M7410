clear;
clc;

%Generate 300 points
[x, y, z, MU, ST] = genPt(300);

%Compute ranges to mean coordinates and sigma
R = sqrt((-100 - x).^2 + (230 - y).^2 + (135 - z).^2);
sigma = sqrt(0.2^2 + 0.3^2 + 0.1^2); 

%Plot 3-D figure
figure(1);
hold on;
grid on;
view(3);
title('Scatter plot', 'FontSize', 15);
xlabel('X',  'FontSize', 10);
ylabel('Y',  'FontSize', 10);
zlabel('Z',  'FontSize', 10);
scatter3(x(R < sigma), y(R < sigma), z(R < sigma), 'filled', 'b');
scatter3(x(R > sigma), y(R > sigma), z(R > sigma), 'filled', 'r');

%Generate 3000, 30000, 300000, and 3000000 points
[x2, y2, z2, MU2, ST2] = genPt(3000);
[x3, y3, z3, MU3, ST3] = genPt(30000);
[x4, y4, z4, MU4, ST4] = genPt(300000);
[x5, y5, z5, MU5, ST5] = genPt(3000000);

%Plot 2-D figures
%mean X
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and mean X', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Mean X',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [MU(1), MU2(1), MU3(1), MU4(1), MU5(1)], '-r*');
axis([0, 3000000, -100.1, -99.9]);

%mean Y
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and mean Y', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Mean Y',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [MU(2), MU2(2), MU3(2), MU4(2), MU5(2)], '-r*');
axis([0, 3000000, 229.9, 230.1]);

%mean Z
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and mean Z', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Mean Z',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [MU(3), MU2(3), MU3(3), MU4(3), MU5(3)], '-r*');
axis([0, 3000000, 134.9, 135.1]);

%standard deviation X
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and std X', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Standard deviation X',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [ST(1), ST2(1), ST3(1), ST4(1), ST5(1)], '-r*');
axis([0, 3000000, 0.18, 0.22]);

%standard deviation Y
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and std Y', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Standard deviation Y',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [ST(2), ST2(2), ST3(2), ST4(2), ST5(2)], '-r*');
axis([0, 3000000, 0.28, 0.32]);

%standard deviation Z
figure('position', [0, 0, 400, 800])
hold on;
grid on;
title('Relationship between number of points and std Z', 'FontSize', 15);
xlabel('Number of points',  'FontSize', 10);
ylabel('Standard deviation Z',  'FontSize', 10);
plot([300, 3000, 30000, 300000, 3000000], [ST(3), ST2(3), ST3(3), ST4(3), ST5(3)], '-r*');
axis([0, 3000000, 0.08, 0.12]);
