import matplotlib.pyplot as plt
import numpy as np

def build_gp_model(train_x, y, test_x, l_scale, max_cov, std_y):
    cov_matrix = compute_cov_matrix(np.concatenate((train_x, test_x)), l_scale, max_cov, std_y)

    cut = len(train_x)
    cov_test_with_data = cov_matrix[cut:,:cut]
    cov_data_with_data = cov_matrix[:cut,:cut]
    cov_test_with_test = cov_matrix[cut:,cut:]

    pred_ys = np.matmul(cov_test_with_data, np.matmul(np.linalg.inv(cov_data_with_data), y))

    var_ys = cov_test_with_test - np.matmul(cov_test_with_data,
        np.matmul(np.linalg.inv(cov_data_with_data), cov_test_with_data.T))
    var_ys = np.diag(var_ys) # off diagonals are the covariance which we don't care about
    return pred_ys, var_ys

def compute_cov_matrix(x, length_scale, variance, train_err):
    xCol = np.expand_dims(x, 1)
    xRow = np.expand_dims(x, 0)
    # Squared exponential
    basic_cov = variance * np.exp(-(xRow - xCol)**2/(2*length_scale))
    # Account for uncertainty on the data
    data_err_cov = np.identity(len(x)) * train_err**2
    return (basic_cov + data_err_cov)

def plot(x, y, err, all_x, pred_ys, var_ys):
    _, ax = plt.subplots()
    ax.errorbar(x, y, yerr=err, ls="", marker=".", label="Training")
    l = ax.plot(all_x, pred_ys, label="GP Function")
    ax.fill_between(all_x, pred_ys - np.sqrt(var_ys), pred_ys + np.sqrt(var_ys), alpha=0.2, color = l[0].get_color())
    ax.legend()
    return ax
