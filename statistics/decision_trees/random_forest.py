import numpy as np
import scipy.stats

from tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees, n_cores=1, tree_config=None):

        default_tree_config = dict(cut_dim="random_best")
        tree_config = {**(tree_config or {}), **default_tree_config}

        self.trees = [DecisionTree(**tree_config) for i in range(n_trees)]
        self.n_cores = n_cores

    def fit(self, trainX, trainY):
        for t in self.trees:
            s = np.random.randint(0, len(trainX), len(trainX))
            t.fit(trainX[s], trainY[s])

    def predict(self, testX):
        preds = [t.predict(testX) for t in self.trees]
        # This is absurdly slow...
        # https://github.com/scipy/scipy/issues/1432
        return scipy.stats.mode(np.array(preds))[0][0]
