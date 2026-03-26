import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
from .utils import entropy_loss

class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 1000, learning_rate: float = 0.01):
        """
        TODO: Implement the training part
        """
        n_samples = X_train.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples
        
        for learner in self.learners:
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32)
            sample_weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float32)
            
            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)
            
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = learner(X_tensor).squeeze()
                critics = entropy_loss(outputs, y_tensor)
                weighted_loss = (sample_weights_tensor * critics).mean()
                weighted_loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                preds = learner(X_tensor).squeeze().numpy()
                preds = np.where(preds >= 0.5, 1, 0)
    
            incorrect = (preds != y_train)
            error = np.sum(self.sample_weights * incorrect) / np.sum(self.sample_weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)  # avoid division by zero
            
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            
            y_pm1 = 2 * y_train - 1
            preds_pm1 = 2 * preds - 1
            
            self.sample_weights *= np.exp(-alpha * y_pm1 * preds_pm1)
            self.sample_weights /= np.sum(self.sample_weights)
        
        
    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the prediction
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        all_class_preds = []
        all_prob_preds = []
        for learner in self.learners:
            with torch.no_grad():
                outputs = learner(X_tensor).squeeze().numpy()
                probs = outputs  # 機率
                preds = (probs >= 0.5).astype(int)  # 類別
                all_class_preds.append(preds)
                all_prob_preds.append(probs)
        return np.array(all_class_preds), np.array(all_prob_preds)
        

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        importance = np.zeros(self.learners[0].fc1.in_features)
        for alpha, learner in zip(self.alphas, self.learners):
            importance += alpha * np.abs(learner.fc1.weight.detach().numpy()).sum(axis=0)
        return importance / np.sum(importance)
       
