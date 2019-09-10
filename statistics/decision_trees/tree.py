import numpy as np
import scipy.optimize, scipy.stats


class DecisionTree:
    eps = 1e-6

    def __init__(self, max_depth=None, min_in_leaf=None, cut_dim="best", _depth=1):
        assert cut_dim in ["best", "random", "random_best"]
        self.n_features = None
        self.n_classes = None

        self.trainX = None
        self.trainY = None

        self.cut_val = None
        self.class_to_cut = None

        self.left = None
        self.right = None

        self.leaf_value = None
        self.depth = _depth
        self.max_depth = max_depth
        self.cut_dim = cut_dim
        self.min_in_leaf = min_in_leaf
        self.random_best_n = lambda: int(np.sqrt(self.n_features)) + 1

    def _child_tree(self):
        return DecisionTree(
            max_depth=self.max_depth,
            cut_dim=self.cut_dim,
            min_in_leaf=self.min_in_leaf,
            _depth=self.depth + 1,
        )

    def predict(self, testX):
        assert testX.shape[1] == self.n_features
        res = np.zeros(len(testX), dtype=self.trainY.dtype)

        # If we are a leaf value, return the prediction
        if self.leaf_value is not None:
            return np.full(len(testX), self.leaf_value, dtype=self.trainY.dtype)

        # Else ask the children for their predictions
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

        # Terminations conditions - note each leaf just returns a single value.
        # More advanced trees might return probabilities, but we keep it simple
        # The class is pure
        if self.n_classes == 1:
            self.leaf_value = self.trainY[0]
            return
        # We have reached the max depth, or the min number of objects
        elif self.depth == self.max_depth or len(self.trainY) == self.min_in_leaf:
            self.leaf_value = scipy.stats.mode(self.trainY)[0][0]
            return

        # Find which feature to cut on, and where to cut
        self.class_to_cut, self.cut_val = self._find_cut()

        # Now we just train the left tree with the values < the split
        self.left = self._child_tree()
        in_left_tree = self.trainX[:, self.class_to_cut] < self.cut_val
        self.left.fit(self.trainX[in_left_tree], self.trainY[in_left_tree])

        # And the right tree with the values >= the split
        self.right = self._child_tree()
        in_right_tree = np.logical_not(in_left_tree)
        self.right.fit(self.trainX[in_right_tree], self.trainY[in_right_tree])

    def _find_cut(self):
        # Compute the max gini gain of making a cut on each feature
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

        # Various strategies to choose between the cuts
        # For a single decision tree, I think you usually just want the best cut.
        # Allowing some randomness is valuable for the forest methods though.
        if self.cut_dim == "best":
            class_to_cut = np.argmin(G_scores["score"])
        elif self.cut_dim == "random":
            class_to_cut = np.random.choice(self.n_features)
        elif self.cut_dim == "random_best":
            class_to_cut = np.random.choice(
                np.argsort(G_scores["score"])[: self.random_best_n()]
            )
        else:
            raise Exception(f"Unknown cut_dim: {self.cut_dim}")

        return class_to_cut, G_scores["cut"][class_to_cut]

    # Nice blog post on gini gain/impurity
    # https://victorzhou.com/blog/gini-impurity/
    # We return the negative as scipy.optimize looks for the minimum
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
