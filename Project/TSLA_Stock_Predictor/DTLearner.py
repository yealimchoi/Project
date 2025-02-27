import numpy as np

class DTLearner:
    def __init__(dec_tree, leaf_size=1, verbose=False):
        dec_tree.leaf_size = leaf_size
        dec_tree.verbose = verbose
        dec_tree.tree = None

    def add_evidence(dec_tree, data_x, data_y):
        dec_tree.tree = dec_tree._build_tree(data_x, data_y)

    def _build_tree(dec_tree, X, Y):
        """
        Recursively builds the decision tree using correlation-based feature selection.
        """
        if X.shape[0] <= dec_tree.leaf_size or np.all(Y == Y[0]):
            return np.array([[-1, np.mean(Y), np.nan, np.nan]])

        # Normalize features
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Compute correlation per feature (Fix: Correct dimension handling)
        correlations = np.array([np.corrcoef(X_norm[:, i], Y)[0, 1] for i in range(X_norm.shape[1])])

        best_feature = np.argmax(np.abs(correlations))  # Use absolute value for best split
        split_val = np.percentile(X[:, best_feature], 50)  # 50th percentile split

        left_mask = X[:, best_feature] < split_val
        right_mask = X[:, best_feature] >= split_val

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(Y), np.nan, np.nan]])

        left_tree = dec_tree._build_tree(X[left_mask], Y[left_mask])
        right_tree = dec_tree._build_tree(X[right_mask], Y[right_mask])

        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(dec_tree, points):
        return np.array([dec_tree._traverse_tree(point, 0) for point in points])

    def _traverse_tree(dec_tree, point, node_idx):
        node = dec_tree.tree[node_idx]
        if int(node[0]) == -1:
            return node[1]
        elif point[int(node[0])] <= node[1]:
            return dec_tree._traverse_tree(point, node_idx + int(node[2]))
        else:
            return dec_tree._traverse_tree(point, node_idx + int(node[3]))

    def author(dec_tree):
        return "your_gt_username"

    def study_group(dec_tree):
        return "your_study_group"
