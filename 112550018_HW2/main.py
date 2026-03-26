import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ):
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        m, n = inputs.shape
        self.weights = np.zeros(n)
        self.intercept = 0.0
        targets = np.array(targets)
        for _ in range(self.num_iterations):
            z = np.dot(inputs, self.weights) + self.intercept
            h = self.sigmoid(z)
            gradient = np.dot(inputs.T, (h - targets)) / m
            intercept_gradient = np.sum(h - targets) / m
            self.weights -= self.learning_rate * gradient
            self.intercept -= self.learning_rate * intercept_gradient
        return None

    def predict(self, inputs: npt.NDArray[np.float64]) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        m = inputs.shape[0]
        X = np.hstack((np.ones((m, 1)), inputs))
        probs = self.sigmoid(np.dot(X, np.hstack((self.intercept, self.weights))))
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """

    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ):
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        S0 = np.dot((X0 - self.m0).T, (X0 - self.m0))
        S1 = np.dot((X1 - self.m1).T, (X1 - self.m1))
        self.sw = S0 + S1

        self.sb = np.outer(self.m1 - self.m0, self.m1 - self.m0)

        self.w = np.linalg.inv(self.sw).dot(self.m1 - self.m0)

        self.slope = (self.w[1]) / (self.w[0])  # For plotting decision boundary
        return None

    def predict(self, inputs: npt.NDArray[np.float64]) -> t.Sequence[t.Union[int, bool]]:

        proj = np.dot(inputs, self.w)
        mean0 = np.dot(self.m0, self.w)
        mean1 = np.dot(self.m1, self.w)
        dist0 = np.abs(proj - mean0)
        dist1 = np.abs(proj - mean1)
        preds = (dist1 < dist0).astype(int)
        return preds

    def plot_projection(self, x, y_true, y_pred):

        w = self.w
        slope_proj = self.slope
        center = (self.m0 + self.m1) / 2
        intercept_proj = center[1] - slope_proj * center[0]

        # w0*x + w1*y = threshold  =>  y = (-w0/w1)x + (threshold/w1)
        threshold = (np.dot(self.m0, self.w) + np.dot(self.m1, self.w)) / 2
        slope_db = -w[0] / w[1]
        intercept_db = threshold / w[1]

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x_vals = np.linspace(x_min, x_max, 200)

        plt.figure(figsize=(8, 6))
        plt.plot(
            x_vals,
            slope_proj
            * x_vals
            + intercept_proj,
            color='gray',
            linestyle='--',
            label=f'Projection line (slope={
                slope_proj:.2f}, intercept={
                intercept_proj:.2f})')
        plt.plot(
            x_vals,
            slope_db * x_vals + intercept_db,
            color='blue',
            label=f'Decision boundary (slope={
                slope_db:.2f}, intercept={
                intercept_db:.2f})')

        for i in range(len(x)):
            marker = 'o' if y_true[i] == 0 else '^'
            color = 'green' if y_true[i] == y_pred[i] else 'red'
            plt.scatter(x[i, 0], x[i, 1], marker=marker, color=color, edgecolor='k', s=80)

        plt.xlabel('Feature 27')
        plt.ylabel('Feature 30')
        plt.title(f'Projection line: slope={slope_proj:.4f}, intercept={intercept_proj:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fld_projection.png')
        plt.close()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    return np.sum(y_trues == y_preds) / len(y_trues)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=2e-4,  # You can modify the parameters as you want
        num_iterations=10000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test, y_pred)


if __name__ == '__main__':
    main()
