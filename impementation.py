import numpy as np
from collections import Counter
import numpy as np

class BallTreeNode:
    def __init__(self, points, indices, radius=0, center=None, left=None, right=None):
        self.points = points           # Точки, содержащиеся в узле
        self.indices = indices         # Индексы точек в исходном наборе данных
        self.radius = radius           # Радиус гиперсферы
        self.center = center           # Центр гиперсферы
        self.left = left               # Левый подузел
        self.right = right             # Правый подузел


class BallTree:
    def __init__(self, X, leaf_size=40):
        self.X = X                     # Входные данные
        self.leaf_size = leaf_size     # Максимальное количество точек в листе
        self.root = self._build_tree(np.arange(X.shape[0]))

    def _build_tree(self, indices):
        """Рекурсивное построение дерева."""
        points = self.X[indices]
        center = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        # Условие остановки
        if len(indices) <= self.leaf_size:
            return BallTreeNode(points, indices, radius, center)
        
        # Расчет расстояний до центра и разделение
        distances = np.linalg.norm(points - center, axis=1)
        median_idx = np.argsort(distances)[len(distances) // 2]
        
        left_indices = indices[distances <= distances[median_idx]]
        right_indices = indices[distances > distances[median_idx]]
        
        left_node = self._build_tree(left_indices)
        right_node = self._build_tree(right_indices)
        
        return BallTreeNode(points, indices, radius, center, left_node, right_node)

    def query(self, point, k):
        """Поиск k ближайших соседей для одной точки."""
        neighbors = []
        self._search(self.root, point, neighbors, k)
        neighbors = sorted(neighbors, key=lambda x: x[0])[:k]
        return [idx for _, idx in neighbors]

    def _search(self, node, point, neighbors, k):
        """Рекурсивный поиск ближайших соседей."""
        if node is None:
            return
        
        # Рассчитаем расстояние от точки до центра узла
        dist_to_center = np.linalg.norm(point - node.center)
        
        # Если текущий узел — лист, проверяем все точки в нем
        if node.left is None and node.right is None:
            for p, idx in zip(node.points, node.indices):
                dist_to_point = np.linalg.norm(point - p)
                neighbors.append((dist_to_point, idx))
            return
        
        # Проверяем левый и правый узел
        if dist_to_center - node.radius <= 0:
            self._search(node.left, point, neighbors, k)
        if dist_to_center + node.radius >= 0:
            self._search(node.right, point, neighbors, k)


class KNN:
    def __init__(self, k=3, task="classification", leaf_size=40):
        """
        :param k: Количество ближайших соседей.
        :param task: Задача - "classification" (классификация) или "regression" (регрессия).
        :param leaf_size: Максимальное количество точек в листе BallTree.
        """
        self.k = k
        self.task = task
        self.leaf_size = leaf_size
        self.tree = None
        self.y = None

    def fit(self, X, y):
        """Обучение модели kNN с использованием BallTree."""
        self.tree = BallTree(X, self.leaf_size)
        self.y = y

    def predict(self, X):
        """Предсказания для набора точек X."""
        predictions = []
        for x in X:
            neighbors = self.tree.query(x, self.k)
            neighbor_labels = self.y[neighbors]
            if self.task == "classification":
                # Возвращаем наиболее часто встречающийся класс
                predictions.append(np.bincount(neighbor_labels).argmax())
            elif self.task == "regression":
                # Возвращаем среднее значение соседей
                predictions.append(np.mean(neighbor_labels))
            else:
                raise ValueError(f"Unknown task: {self.task}. Use 'classification' or 'regression'.")
        return np.array(predictions)


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, task="classification"):
        self.lr = lr
        self.n_iters = n_iters
        self.task = task
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.task == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

            self.weights = np.zeros((n_classes, n_features))
            self.bias = np.zeros(n_classes)

            # One-vs-Rest 
            for idx, cls in enumerate(self.classes_):
                y_binary = np.where(y == cls, 1, 0)

                for _ in range(self.n_iters):
                    linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]
                    y_predicted = self._sigmoid(linear_model)

                    dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
                    db = (1 / n_samples) * np.sum(y_predicted - y_binary)

                    self.weights[idx] -= self.lr * dw
                    self.bias[idx] -= self.lr * db

        elif self.task == "regression":
            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iters):
                linear_model = np.dot(X, self.weights) + self.bias
                y_predicted = linear_model

                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias

        if self.task == "classification":
            y_predicted = self._sigmoid(linear_model)
            return np.argmax(y_predicted, axis=1)
        elif self.task == "regression":
            return linear_model

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Узел дерева решений.
        feature: индекс признака для разделения (если это не листовой узел)
        threshold: порог разделения (если это не листовой узел)
        left: левый дочерний узел
        right: правый дочерний узел
        value: значение предсказания (только для листового узла)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Проверка, является ли узел листовым."""
        return self.value is not None



class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.tree = None

    def fit(self, X, y):
        """Обучение дерева решений."""
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """Предсказание для набора данных."""
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        """Рекурсивное построение дерева."""
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            if self.task == "classification":
                most_common_class = np.argmax(np.bincount(y))
                return TreeNode(value=most_common_class)
            else:
                mean_value = np.mean(y)
                return TreeNode(value=mean_value)

        best_gain = -1
        split_idx, split_threshold = None, None

        current_metric = self._gini(y) if self.task == "classification" else self._mse(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y, right_y = y[left_mask], y[right_mask]
                left_metric = self._gini(left_y) if self.task == "classification" else self._mse(left_y)
                right_metric = self._gini(right_y) if self.task == "classification" else self._mse(right_y)

                gain = current_metric - (len(left_y) / n_samples) * left_metric - (len(right_y) / n_samples) * right_metric

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        if best_gain == -1:
            if self.task == "classification":
                most_common_class = np.argmax(np.bincount(y))
                return TreeNode(value=most_common_class)
            else:
                mean_value = np.mean(y)
                return TreeNode(value=mean_value)

        left_mask = X[:, split_idx] <= split_threshold
        right_mask = X[:, split_idx] > split_threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=split_idx, threshold=split_threshold, left=left_child, right=right_child)

    def _traverse_tree(self, x, node):
        """Прохождение по дереву для одной выборки."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _gini(self, y):
        """Вычисление критерия Джини для классификации."""
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def _mse(self, y):
        """Вычисление среднего квадратичного отклонения для регрессии."""
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=100, sample_size=0.8, task="classification"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.task = task
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, int(n_samples * self.sample_size), replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(max_depth=self.max_depth, task=self.task)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "classification":
            return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)
        elif self.task == "regression":
            return np.mean(tree_predictions, axis=0)

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, task="regression"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task = task
        self.models = []

    def fit(self, X, y):
        self.models = []
        y_pred = np.mean(y) if self.task == "regression" else np.zeros(len(y))

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTree(max_depth=self.max_depth, task="regression")
            tree.fit(X, residuals)
            self.models.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred if self.task == "regression" else (y_pred > 0.5).astype(int)
