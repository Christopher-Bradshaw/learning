
def write(arr, filename):
    assert len(arr.shape) == 3 and arr.shape[2] == 3
    x_len = arr.shape[0]
    y_len = arr.shape[1]

    f = open(filename, "w")

    f.write("P3\n")
    f.write(f"{x_len} {y_len}\n")
    f.write("255\n")


    for j in range(y_len):
        for i in range(x_len):
            p = (arr[i,j] * 255).astype(int)
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
