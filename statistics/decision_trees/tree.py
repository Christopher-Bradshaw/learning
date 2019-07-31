import numpy as np
import scipy.optimize


class DecisionTree:
    eps = 1e-3

    def __init__(self):
        self.n_features = None
        self.n_classes = None

        self.trainX = None
        self.trainY = None

        self.cut_val = None
        self.class_to_cut = None

        self.left = None
        self.right = None

        self.leaf_value = None

    def predict(self, testX):
        assert testX.shape[1] == self.n_features
        res = np.zeros(len(testX), dtype=self.trainY.dtype)

        if self.leaf_value is not None:
            return np.full(len(testX), self.leaf_value, dtype=self.trainY.dtype)

        in_left_tree = testX[:, self.class_to_cut] < self.cut_val
        in_right_tree = np.logical_not(in_left_tree)

        res[in_left_tree] = self.left.predict(testX[in_left_tree])
        res[in_right_tree] = self.right.predict(testX[in_right_tree])
        return res

    def fit(self, trainX, trainY):
        assert len(trainY.shape) == 1
        self.n_features = trainX.shape[1]
        self.n_classes = len(np.unique(trainY))

        self.trainX = trainX
        self.trainY = trainY

        if self.n_classes == 1:
            self.leaf_value = trainY[0]
            return

        self.class_to_cut, self.cut_val = self._find_cut()

        self.left = DecisionTree()
        in_left_tree = self.trainX[:, self.class_to_cut] < self.cut_val
        self.left.fit(self.trainX[in_left_tree], self.trainY[in_left_tree])
        self.right = DecisionTree()
        in_right_tree = np.logical_not(in_left_tree)
        self.right.fit(self.trainX[in_right_tree], self.trainY[in_right_tree])

    def _find_cut(self):
        G_scores = np.zeros(
            (self.n_features,), dtype=[("score", np.float32), ("cut", np.float32)]
        )
        for i in range(self.n_features):
            x = self.trainX[:, i]
            y = self.trainY
            best_cut, G_score, _, _ = scipy.optimize.brute(
                self._compute_negative_gini_gain,
                ranges=[(np.min(x) + self.eps, np.max(x) - self.eps)],
                args=(x, y),
                full_output=True,
                finish=None,
            )
            G_scores[i] = (G_score, best_cut)
        class_to_cut = np.argmin(G_scores["score"])
        return class_to_cut, G_scores["cut"][class_to_cut]

    def _compute_negative_gini_gain(self, cut, x, y):
        assert len(x.shape) == 1 and len(y.shape) == 1
        assert np.min(x) < cut < np.max(x)
        below_cut = x < cut
        above_cut = np.logical_not(below_cut)
        G_before = self._compute_gini_impurity(y)
        G_l = np.count_nonzero(below_cut) * self._compute_gini_impurity(y[below_cut])
        G_r = np.count_nonzero(above_cut) * self._compute_gini_impurity(y[above_cut])
        return -(G_before - (G_l + G_r) / len(y))

    def _compute_gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        G = 0
        for c in counts:
            G += c / len(y) * (1 - c / len(y))
        return G
