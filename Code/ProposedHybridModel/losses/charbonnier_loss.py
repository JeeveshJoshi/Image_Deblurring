import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (a robust L1-like loss).
    L(x, y) = sqrt((x - y)^2 + epsilon^2)
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        return torch.mean(torch.sqrt((prediction - target)**2 + self.epsilon**2))