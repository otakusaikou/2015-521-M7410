import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# plot data
x = np.array([i for i in range(20, 81, 5)])
y = np.array([.02, .04, .07, .11, .2, .23, .32, .35, .46, .5, .62, .65, .79])

plt.figure(figsize=(8, 8))
# plt.plot(x, y, 'ro')
plt.title("Relationship between factor A and D", size=20)
plt.xlabel('Age', size=15)
plt.ylabel('Disease Rate', size=15)

plt.axis([0, 100, 0, 1])

# calculate parameters of math model
A = np.matrix([[20, 1],
               [25, 1],
               [30, 1],
               [35, 1],
               [40, 1],
               [45, 1],
               [50, 1],
               [55, 1],
               [60, 1],
               [65, 1],
               [70, 1],
               [75, 1],
               [80, 1]])

L = np.matrix([[0.02],
               [0.04],
               [0.07],
               [0.11],
               [0.20],
               [0.23],
               [0.32],
               [0.35],
               [0.46],
               [0.50],
               [0.62],
               [0.65],
               [0.79]])

N = (A.T * A)
W = (A.T * L)

X = N.I * W
V = A * X - L

sig0 = sqrt((V.T * V) / (A.shape[0] - A.shape[1]))
Q = N.I
sigX = sig0 * sqrt(Q[0, 0])
sigY = sig0 * sqrt(Q[1, 1])

# plot best fit line
slope, intercept = X[0, 0], X[1, 0]
x2 = np.array([i for i in range(20, 101, 20)])
plt.plot(x2, x2 * slope + intercept, 'b')


plt.plot(x2, x2 * slope + intercept + sig0, 'r--')
plt.annotate(
    "$D'=aA+b+\sigma_0$", xy=(60, 60 * slope + intercept + sig0),
    xytext=(-80, 80), textcoords="offset points", va="center",
    size=15,
    arrowprops=dict(arrowstyle="->"))

plt.plot(x2, x2 * slope + intercept - sig0, 'r--')
plt.annotate(
    "$D''=aA+b-\sigma_0$", xy=(80, 80 * slope + intercept - sig0),
    xytext=(-10, -80), textcoords="offset points", va="center",
    size=15,
    arrowprops=dict(arrowstyle="->"))

for i in range(len(x)):
    if y[i] > x[i] * slope + intercept + sig0 or y[i] < x[i] * slope + intercept - sig0:
        plt.plot(x[i], y[i], 'bo')
    else:
        plt.plot(x[i], y[i], 'ro')

plt.show()

print sig0, sigX, sigY
