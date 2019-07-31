from tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees, tree_config=None):
        default_tree_config = dict(cut_dim="random_best")
        tree_config = {**(tree_config or {}), **default_tree_config}

        self.trees = [DecisionTree(**tree_config) for i in n_trees]

    def fit(self, trainX, trainY):
        for tree in self.trees:
            tree.fit(trainX, trainY)

    def predict(self, testX):
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(testX))
        return preds
