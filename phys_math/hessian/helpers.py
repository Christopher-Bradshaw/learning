import numpy as np
import matplotlib.pyplot as plt

def taylor_1d(a, val, der, der2):
    return val + der * a + 0.5 * der2 * a**2

def taylor_2d(grid, x0, val, grad, hess):
    grid = np.reshape(grid, (2, -1))
    res = np.zeros_like(grid[0])
    for i, a in enumerate(grid.T):
        a = a - x0
        res[i] = val + np.dot(grad, a) + 0.5 * np.matmul(a.T, np.matmul(hess, a))
    res = np.reshape(res, (200, 200))
    return res

def plot_hess(grad, hess):
    lim = 2
    print_hess_info(hess)

    x = np.linspace(-lim, lim, num=200)
    y = np.linspace(-lim, lim, num=200)
    grid = np.array(np.meshgrid(x, y))

    x0 = np.array([0, 0])

    fig, ax = plt.subplots()
    fig.set_size_inches((fig.get_size_inches()[1], fig.get_size_inches()[1]))

    cs = ax.contour(grid[0], grid[1], taylor_2d(
        grid, x0, 0, grad, hess), cmap="Greys")
    ax.clabel(cs, fontsize="x-small")
    ax.scatter(x0[0], x0[1])

def print_hess_info(hess):
    eigenvals, eigenvecs = np.linalg.eig(hess)
    print(eigenvals, eigenvecs)
    if np.all(eigenvals > 0):
        print("Hessian is PD")
    elif np.all(eigenvals >= 0):
        print("Hessian is PSD")
    elif np.all(eigenvals < 0):
        print("Hessian is ND")
    elif np.all(eigenvals <= 0):
        print("Hessian is NSD")
    else:
        print("Hessian is indefinite")
