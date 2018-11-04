import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    dims = 2
    # Rule 1) sqrt(v1**2 + v2**2 + ...) == 2
        # Or v1**2 + v2**2 + ... == 4
    # Rule 2) sqrt( (v1-o1)**2 + ... ) >= 2
    # Rule 3) -2 <= vi <= 2

    points = []

    consec_failures = 0
    while consec_failures < 10000:
        new_point = 2*_point_on_hypersphere(dims)
        for p in points:
            if np.linalg.norm(new_point - p) < 2:
                break
        else: # if we didn't break
            points.append(new_point)
            consec_failures = 0
            continue
        consec_failures += 1

    points = np.array(points)
    print(len(points))

    _plot2d(points)


def _plot2d(points):
    _, ax = plt.subplots()

    ax.scatter(0, 0, color="red")
    circle1 = plt.Circle((0,0), 1, edgecolor="red", facecolor="none")
    ax.add_artist(circle1)

    for p in points:
        ax.scatter(p[0], p[1], color="blue")
        circle1 = plt.Circle((p[0],p[1]), 1, edgecolor="blue", facecolor="none")
        ax.add_artist(circle1)

    ax.set(
            xlim=(-3, 3),
            ylim=(-3, 3),
    )
    plt.show()

# http://mathworld.wolfram.com/HyperspherePointPicking.html
def _point_on_hypersphere(dim):
    x = norm.rvs(size = dim)
    p = 1/np.linalg.norm(x) * x
    return p



if __name__ == "__main__":
    main()
