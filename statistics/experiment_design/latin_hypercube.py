import numpy as np
import pyDOE as p
import matplotlib.pyplot as plt

# Create 5 strata for a single variable and randomly select from each strata
lh = p.lhs(1, samples=5)
# e.g.
# [[0.719] [0.468] [0.385] [0.857] [0.143]]

lh = p.lhs(2, samples=3)
# e.g.
# [[0.796 0.524]  [0.616 0.078]  [0.129 0.863]] (or 3,2  2,1, 1,3)
# [[0.389 0.474]  [0.029 0.828]  [0.755 0.289]] (or 2,2  1,3  3,1)
# Ignoring ordering, there are 3! possible pairings

# By default we have been sampling randomly but we can also
num_samples = 20

for crit in ["center", "centermaximin", "maximin", "corr"]:
    lh = p.lhs(2, samples=num_samples, criterion=crit)

    _, ax = plt.subplots()

    ax.scatter(
            [i[0] for i in lh],
            [i[1] for i in lh],
    )
    for div in np.linspace(0, 1, num=num_samples+1):
        ax.axvline(div)
        ax.axhline(div)
    ax.set(title=crit)
plt.show()
