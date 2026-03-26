"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self, lr=1e-4):
        self.lr = lr
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        X_T = X_ext.T
        inv = np.linalg.pinv(X_T @ X_ext)
        w_all = inv @ X_T @ y
        self.weights = w_all[:-1]
        self.intercept = w_all[-1]

    def predict(self, X):
        """Question4
        Complete this function
        """
        y_pred = X @ self.weights + self.intercept
        return y_pred


class LinearRegressionGradientdescent(LinearRegressionBase):
    def __init__(self, lr=1e-4):
        super().__init__(lr)

    def fit(self, X, y, learning_rate, epochs: int):
        """Question2
        Complete this function
        """
        y = y.reshape(-1)
        self.lr = learning_rate
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0
        losses, lr_history = [], []
        decay_rate = 1e-4
        # theta = theta - (alpha / (1 + decay_rate * epoch)) * dJ(theta)/dtheta
        for epoch in range(epochs):
            y_pred = X @ self.weights + self.intercept
            loss = compute_mse(y_pred, y)
            grad_w = (2 / n_samples) * (X.T @ (y_pred - y))
            grad_b = (2 / n_samples) * np.sum(y_pred - y)

            # self.weights -= self.lr * grad_w
            # self.intercept -= self.lr * grad_b

            LR = self.lr / (1 + decay_rate * epoch)

            self.weights -= LR * grad_w
            self.intercept -= LR * grad_b

            losses.append(loss)
            lr_history.append(LR)

            if epoch % 10000 == 0:
                logger.info(f"EPOCH {epoch}, {loss=:.4f}, {LR=:.6f}")
        return losses, lr_history

    def predict(self, X):
        """Question4
        Complete this
        """
        y_pred = X @ self.weights + self.intercept
        return y_pred


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv("./train.csv")  # Load training data
    test_df = pd.read_csv("./test.csv")  # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f"{LR_CF.weights=}, {LR_CF.intercept=:.4f}")

    # Test the closed form
    # from sklearn.linear_model import LinearRegression
    # sk_lr = LinearRegression()
    # sk_lr.fit(train_x, train_y)
    # sk_pred = sk_lr.predict(test_x)
    # logger.info(f'sklearn weights={sk_lr.coef_}, intercept={sk_lr.intercept_:.4f}')

    # Feature Scailing for GD
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    G_train_x = scaler.fit_transform(train_x)
    G_test_x = scaler.transform(test_x)

    LR_GD = LinearRegressionGradientdescent(lr=1e-4)
    losses, lr_history = LR_GD.fit(
        G_train_x, train_y, learning_rate=2e-4, epochs=100000
    )

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f"{LR_GD.weights=}, {LR_GD.intercept=:.4f}")

    """
    Question3: Plot the learning curve.
    Implement here
    """
    plt.figure(figsize=(10, 6))
    epochs = len(losses)
    plt.plot(losses, label="Train MSE loss")

    mark_epochs = np.arange(0, epochs, 1000)
    plt.scatter(
        mark_epochs,
        [losses[i] for i in mark_epochs],
        color="blue",
        s=30,
        label="Every 1000 Epoch",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("Loss_curve.png")

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(G_test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f"Prediction difference: {y_preds_diff:.4f}")

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f"{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%")


if __name__ == "__main__":
    main()
