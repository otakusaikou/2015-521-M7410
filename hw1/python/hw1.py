import matplotlib.pyplot as plt
from numpy import random, mean, std
from mpl_toolkits.mplot3d import Axes3D, proj3d

def genPt(num):
    #Set error parametrs
    param = [(-100, 0.2), (230, 0.3), (135, 0.1)]

    #Create coordinates 
    x, y, z = map(lambda x: random.normal(x[0], x[1], num), param)

    #Compute mean and standard deviation values
    mx, my, mz = map(lambda x: mean(x), [x, y, z])
    stdx, stdy, stdz = map(lambda x: std(x), [x, y, z])

    return x, y, z, mx, my, mz, stdx, stdy, stdz

def drawScatter(x, y, z, fig, mx, my, mz, sigma):
    #Compute ranges to mean coordinates and sigma
    R = ((mx - x)**2 + (my - y)**2 + (mz - z)**2)**0.5

    #Create figure
    fig = plt.figure(fig, figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(111, projection="3d")

    #Disable scientific notation of z axis
    ax.w_zaxis.get_major_formatter().set_useOffset(False)

    #Set title and labels
    plt.title("Scatter plot", size=25)
    ax.set_xlabel("X", fontsize = 15)
    ax.set_ylabel("Y", fontsize = 15)
    ax.set_zlabel("Z", fontsize = 15)

    #Plot all points
    ax.plot(x[R < sigma], y[R < sigma], z[R < sigma], "8", color = "b", label = "1")
    ax.plot(x[R > sigma], y[R > sigma], z[R > sigma], "8", color = "r", label = "1")

    #Show plot
    plt.show()

def drawFunctionPlot(data, title, ylabel, fig):
    #Create figure
    fig = plt.figure(fig)
    ax = fig.add_subplot(211)

    #Enable grid line
    plt.grid()

    #Disable scientific notation of z axis
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    #Set x axis range
    ax.set_xlim([-100000, 3100000])

    #Set title and labels
    plt.title(title, size=25)
    ax.set_xlabel("Number of points", fontsize = 15)
    ax.set_ylabel(ylabel, fontsize = 15)
     
    #Plot all points and line 
    plt.plot([3*10**i for i in range(2, 7)], data, 'bo')
    plt.plot([3*10**i for i in range(2, 7)], data, 'b-')

    #Show plot
    plt.show()

def main():
    #Generate 300 points
    x, y, z, mx, my, mz, stdx, stdy, stdz = genPt(300)
    sigma = (0.2**2 + 0.3**2 + 0.1**2)**0.5
    
    #Plot 3-D figure
    drawScatter(x, y, z, 0, -100, 230, 135, sigma)

    #Generate 3000, 30000, 300000, and 3000000 points
    mx2, my2, mz2, stdx2, stdy2, stdz2 = genPt(3000)[3:]
    mx3, my3, mz3, stdx3, stdy3, stdz3 = genPt(30000)[3:]
    mx4, my4, mz4, stdx4, stdy4, stdz4 = genPt(300000)[3:]
    mx5, my5, mz5, stdx5, stdy5, stdz5 = genPt(3000000)[3:]

    #Plot 2-D figure
    #Mean values
    drawFunctionPlot([mx, mx2, mx3, mx4, mx5], "Relationship between number of points and mean X", "Mean X", 1)
    drawFunctionPlot([my, my2, my3, my4, my5], "Relationship between number of points and mean Y", "Mean Y", 2)
    drawFunctionPlot([mz, mz2, mz3, mz4, mz5], "Relationship between number of points and mean Z", "Mean Z", 3)

    #Standard deviation
    drawFunctionPlot([stdx, stdx2, stdx3, stdx4, stdx5], "Relationship between number of points and S.D. X", "Standard deviation X", 4)
    drawFunctionPlot([stdy, stdy2, stdy3, stdy4, stdy5], "Relationship between number of points and S.D. Y", "Standard deviation Y", 5)
    drawFunctionPlot([stdz, stdz2, stdz3, stdz4, stdz5], "Relationship between number of points and S.D. Z", "Standard deviation Z", 6)

    #Print mean coordinates and standard deviation values
    #print mx, my, mz, stdx, stdy, stdz
    #print mx2, my2, mz2, stdx2, stdy2, stdz2
    #print mx3, my3, mz3, stdx3, stdy3, stdz3
    #print mx4, my4, mz4, stdx4, stdy4, stdz4
    #print mx5, my5, mz5, stdx5, stdy5, stdz5

    return 0

if __name__ == "__main__":
    main()

