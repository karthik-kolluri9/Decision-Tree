# Defining Decision Tree class with Bayesian probabilities calculation
import numpy as np
import pandas as pd
import collections

# Tree node class to maintain values in left and right node with Bayesian probability
class TreeNode:
    def __init__(self, train_feature=None, bayesian_prob=None, left_node=None, right_node=None, info_gain=None, target_value=None):
        self.train_feature = train_feature
        self.bayesian_prob = bayesian_prob
        self.left_node = left_node
        self.right_node = right_node
        self.info_gain = info_gain
        self.target_value = target_value

# Implementing decision tree classifier using information gain
class CartDecisionTreeClass:
    def __init__(self, minimum_samples_leaf=2, max_depth_size=5):
        self.minimum_samples_leaf = minimum_samples_leaf
        self.max_depth_size = max_depth_size
        self.tree_root = None

    def bestSplit(self, X_train, y_train):
        # Dummy function for best split logic (to be implemented)
        return {'info_gain': 1, 'featureIndex': 0, 'bayesian_prob': 0.5, 'data_left': X_train, 'data_right': X_train}

    def balancingbuildTree(self, X_train, y_train, tree_depth=0):
        rows, cols = X_train.shape
        if rows >= self.minimum_samples_leaf and tree_depth <= self.max_depth_size:
            best = self.bestSplit(X_train, y_train)
            if best['info_gain'] > 0:
                left_node = self.balancingbuildTree(best['data_left'][:, :-1], best['data_left'][:, -1], tree_depth + 1)
                right_node = self.balancingbuildTree(best['data_right'][:, :-1], best['data_right'][:, -1], tree_depth + 1)

                if left_node.bayesian_prob is not None:
                    print(f"Left Node No: {tree_depth} Probability: {left_node.bayesian_prob}")
                if right_node.bayesian_prob is not None:
                    print(f"Right Node No: {tree_depth} Probability: {right_node.bayesian_prob}")

                return TreeNode(
                    train_feature=best["featureIndex"],
                    bayesian_prob=best['bayesian_prob'],
                    left_node=left_node,
                    right_node=right_node,
                    info_gain=best["info_gain"]
                )

        return TreeNode(target_value=collections.Counter(y_train).most_common(1)[0][0])

    def fit(self, X_train, y_train):
        self.tree_root = self.balancingbuildTree(X_train, y_train)

    # Function to calculate probabilities of each node features
    def getProbability(self, X_test, dt_tree):
        if dt_tree.target_value is not None:
            return dt_tree.target_value

        features = X_test[dt_tree.train_feature]

        if features <= dt_tree.bayesian_prob:
            return self.getProbability(X_test, dt_tree.left_node)
        else:
            return self.getProbability(X_test, dt_tree.right_node)

    # Calculate probability on given test features to predict outcome
    def calculateProbability(self, X_test):
        predicted_labels = [self.getProbability(features, self.tree_root) for features in X_test]
        return predicted_labels
