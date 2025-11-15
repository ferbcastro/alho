import torch
import torch.nn as nn

'''
O gamma aumenta o loss para predições difíceis (gamma igual 2 parece bom)
O alpha ajuda a lidar com desbalanceamento entre classes. Por exemplo, se
alpha=0.25 e valor esperado é 0, então seu bce_loss recebe (1-0.25)*bce_loss.
Se o problema faz duas classificações com classe 0 majoritária, logo o valor de
alpha deve ser alto para atribuir erro maior para instâncias de classe 1.
    ~ Fernando
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        """ Focal loss for binary classification. """

        # Compute binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
