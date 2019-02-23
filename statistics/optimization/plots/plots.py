import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def surface_2d(f, lim=None, typ="contour"):
    lim = lim or np.array([-1, 1, -1, 1]) * 2
    x = np.linspace(lim[0], lim[1], num=200)
    y = np.linspace(lim[2], lim[3], num=200)
    grid = np.array(np.meshgrid(x, y))

    res = f(np.reshape(grid, (2, -1))).reshape(grid.shape[1:])

    if typ == "contour":
        fig, ax = plt.subplots()
        fig.set_size_inches((fig.get_size_inches()[1], fig.get_size_inches()[1]))
        cs = ax.contour(grid[0], grid[1], res, cmap="Greys",
                levels=np.geomspace(np.percentile(res, 5), np.percentile(res, 80), num=6))
        ax.clabel(cs, fontsize="x-small")
    elif typ == "image":
        fig, ax = plt.subplots()
        fig.set_size_inches((fig.get_size_inches()[1], fig.get_size_inches()[1]))
        img = ax.imshow(res, cmap="Reds", origin="lower", extent=(x[0], x[-1], y[0], y[-1]))
        fig.colorbar(img)
    elif typ == "surface":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid[0], grid[1], res)
    else:
        raise Exception("Unknown plot typ")

    ax.set(xlim=lim[:2], ylim=lim[2:])

    return ax
