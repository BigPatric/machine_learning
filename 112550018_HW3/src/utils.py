import typing as t
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation in the `intermediate layers` allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
def entropy_loss(outputs, targets):
    epsilon = 1e-10  # to avoid log(0)
    loss = - (targets * torch.log(outputs + epsilon) + (1 - targets) * torch.log(1 - outputs + epsilon)).mean()
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure(figsize=(8, 6))
    for i, y_pred in enumerate(y_preds):
        y_pred = np.array(y_pred).ravel()
        if len(np.unique(y_trues)) == 1:
            continue 
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f'Learner {i+1} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Each Learner')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

def plot_feature_importance(feature_names, importances, fpath='./feature_importance.png'):
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()