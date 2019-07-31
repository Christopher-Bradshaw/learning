import numpy as np
from tree import DecisionTree

t = DecisionTree()

x = np.arange(10)[:, np.newaxis]
y = np.arange(10) > 5

t.fit(x, y)
print(t.predict(x))
