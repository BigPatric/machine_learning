import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
from .utils import entropy_loss

class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """
        TODO: Implement the training part
        """
        for learner in self.learners:
            idx = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap, y_bootstrap = X_train[idx], y_train[idx]
            
            X_tensor = torch.tensor(X_bootstrap, dtype=torch.float32)
            y_tensor = torch.tensor(y_bootstrap, dtype=torch.float32)
            
            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = learner(X_tensor).squeeze()
                loss = entropy_loss(outputs, y_tensor)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        
    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the training part
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        all_class_preds = []
        all_prob_preds = []
        for learner in self.learners:
            with torch.no_grad():
                probs = learner(X_tensor).squeeze().numpy()
                preds = (probs >= 0.5).astype(int)
                all_class_preds.append(preds)
                all_prob_preds.append(probs)
        return np.array(all_class_preds), np.array(all_prob_preds)

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        importance = np.zeros(self.learners[0].fc1.in_features)
        for learner in self.learners:
            importance += np.abs(learner.fc1.weight.detach().numpy()).sum(axis=0)
        return importance / np.sum(importance)