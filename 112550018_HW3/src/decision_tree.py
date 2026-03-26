import numpy as np



class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.n_samples_ = len(y)
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        if depth >= self.max_depth or num_labels == 1 or num_samples == 0:
            leaf_class = self._majority_vote(y)
            return {'type': 'leaf', 'class': leaf_class}
        feature_index, threshold = find_best_split(X, y)
        if feature_index is None:
            leaf_class = self._majority_vote(y)
            return {'type': 'leaf', 'class': leaf_class}
        left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
        gain = information_gain(y, left_y, right_y)
        left_subtree = self._grow_tree(left_X, left_y, depth + 1)
        right_subtree = self._grow_tree(right_X, right_y, depth + 1)
        return {
            'type': 'node',
            'feature_index': feature_index,
            'threshold': threshold,
            'n': len(y),
            'gain': gain,
            'left': left_subtree,
            'right': right_subtree
        }

    def _majority_vote(self, y):
        if(len(y) == 0):
            return 0
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if tree_node['type'] == 'leaf':
            return tree_node['class']
        if x[tree_node['feature_index']] <= tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def compute_feature_importance(self):
        # 使用每個節點的資訊增益乘上節點樣本比例作為貢獻度
        importance = np.zeros(self.n_features_)

        def traverse(node):
            if node['type'] == 'node':
                contrib = (node.get('gain', 0.0)) * (node.get('n', 0) / max(1, self.n_samples_))
                importance[node['feature_index']] += contrib
                traverse(node['left'])
                traverse(node['right'])

        traverse(self.tree)
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return importance


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_idx = X[:, feature_index] <= threshold
    right_idx = X[:, feature_index] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]


# Find the best split for the dataset
def find_best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None
    n_samples, n_features = X.shape
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                continue
            gain = information_gain(y, y[left_indices], y[right_indices])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold

def information_gain(parent, left_child, right_child):
    parent_entropy = entropy(parent)
    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)
    if n_left == 0 or n_right == 0:
        return 0
    child_entropy = (n_left / n) * entropy(left_child) + (n_right / n) * entropy(right_child)
    ig = parent_entropy - child_entropy
    return ig

def entropy(y):
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])