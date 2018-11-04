import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    _, ax = plt.subplots()
    ax.imshow(np.log10(output), cmap="gist_heat")
    plt.axis("off")
    plt.savefig("mandelbrot.png", bbox_inches='tight')
    plt.show()
