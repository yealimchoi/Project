import numpy as np

class RTLearner:
    def __init__(ran_tree, leaf_size=1, verbose=False):
        """
        Initialize Random Tree Learner.
        :param leaf_size: Minimum number of samples to allow in a leaf.
        :param verbose: Debug mode.
        """
        ran_tree.leaf_size = leaf_size
        ran_tree.verbose = verbose
        ran_tree.tree = None

    def add_evidence(ran_tree, data_x, data_y):
        """
        Train the random tree.
        :param data_x: Feature data (NumPy array)
        :param data_y: Target values (NumPy array)
        """
        ran_tree.tree = ran_tree._build_tree(data_x, data_y)

    def _build_tree(ran_tree, X, Y):
        """
        Recursively builds the random tree using random feature selection.
        """
        if X.shape[0] <= ran_tree.leaf_size or np.all(Y == Y[0]):
            return np.array([[-1, np.mean(Y), np.nan, np.nan]])  # Leaf node

        # Select a random feature for splitting
        best_feature = np.random.randint(0, X.shape[1])
        split_val = np.median(X[:, best_feature])  # Split using median

        left_mask = X[:, best_feature] < split_val
        right_mask = X[:, best_feature] >= split_val

        # Ensure valid split: if one side is empty, use mean split instead
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            split_val = np.mean(X[:, best_feature])
            left_mask = X[:, best_feature] < split_val
            right_mask = X[:, best_feature] >= split_val

        # If split is still invalid, return leaf node
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(Y), np.nan, np.nan]])

        left_tree = ran_tree._build_tree(X[left_mask], Y[left_mask])
        right_tree = ran_tree._build_tree(X[right_mask], Y[right_mask])

        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(ran_tree, points):
        """
        Predict target values for test data.
        :param points: Test feature data.
        :return: Predicted target values.
        """
        return np.array([ran_tree._traverse_tree(point, 0) for point in points])

    def _traverse_tree(ran_tree, point, node_idx):
        """
        Recursively traverse the tree to make a prediction.
        """
        node = ran_tree.tree[node_idx]
        if int(node[0]) == -1:  # Leaf node
            return node[1]
        elif point[int(node[0])] <= node[1]:  # Go left
            return ran_tree._traverse_tree(point, node_idx + int(node[2]))
        else:  # Go right
            return ran_tree._traverse_tree(point, node_idx + int(node[3]))

    def author(ran_tree):
        return "your_gt_username"

    def study_group(ran_tree):
        return "your_study_group"
