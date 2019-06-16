import numpy as np
import matplotlib.pyplot as plt

def main():
    nx, ny = 200, 100

    img = np.zeros((ny, nx, 3))
    img[2] = 0.2

    plt.imshow(img)
    plt.show()




if __name__ == "__main__":
    main()
