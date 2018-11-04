import numpy as np
import george
import matplotlib.pyplot as plt

# Create some data that follows a sin function with some noise
np.random.seed(12)
x = np.sort(10 * np.random.random(15))
y_err = 0.0002 * np.ones_like(x)
y = np.sin(x) + y_err * np.random.normal(0, 1, size=len(x))


# GP prediction?
kernel = np.var(y) * george.kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, y_err)

x_pred = np.linspace(0, 10, 500)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

# Plot and see the true function, the data and the gp
_, ax = plt.subplots()
ax.plot(x_pred, pred, "blue", lw=1.5, alpha=0.5, label="prediction")
ax.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="blue", alpha=0.2)
ax.errorbar(x, y, yerr=y_err, fmt=".k", capsize=0, label="data")
ax.plot(x_pred, np.sin(x_pred), "--g", label="truth")
ax.legend()
plt.show()
