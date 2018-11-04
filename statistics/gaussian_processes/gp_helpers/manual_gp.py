import matplotlib.pyplot as plt
import numpy as np

def build_gp_model(x, y, all_x, covariance_function, l_scale, max_cov, **kwargs):
    cov_matrix = covariance_function(np.concatenate((x, all_x)), l_scale, max_cov, **kwargs)
    print(cov_matrix)

    cut = len(x)
    cov_test_with_data = cov_matrix[cut:,:cut]
    cov_data_with_data = cov_matrix[:cut,:cut]
    cov_test_with_test = cov_matrix[cut:,cut:]

    pred_ys = np.matmul(cov_test_with_data, np.matmul(np.linalg.inv(cov_data_with_data), y))

    var_ys = cov_test_with_test - np.matmul(cov_test_with_data,
        np.matmul(np.linalg.inv(cov_data_with_data), cov_test_with_data.T))
    var_ys = np.diag(var_ys) # off diagonals are the covariance which we don't care about
    return pred_ys, var_ys

def plot(x, y, err, f, all_x, pred_ys, var_ys):
    _, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, y, yerr=err, ls="", marker=".")
    ax.plot(all_x, f(all_x))
    ax.plot(all_x, pred_ys)
    ax.fill_between(all_x, pred_ys - np.sqrt(var_ys), pred_ys + np.sqrt(var_ys), alpha=0.2, color="green")
    ax.set(ylim=(-2, 2))
    return ax
