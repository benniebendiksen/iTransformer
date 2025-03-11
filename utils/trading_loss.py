import torch
import torch.nn as nn
import numpy as np


class TradingBCELoss(nn.Module):
    """
    Custom BCE loss for trading that accounts for shorting strategy
    (shorting versus holding during predicted downturns) and class imbalance.

    When is_shorting=False:
    - We penalize false positives more heavily (precision focus for uptrend predictions)
    - We reduce penalty for false negatives (since not shorting means we just hold)

    Parameters:
    -----------
    is_shorting : bool
        Whether the trading strategy includes shorting (True) or just holding (False) for predicted downtrends

    precision_factor : float
        How much to weight precision for non-shorting strategies (higher values increase precision)
        Only applies when is_shorting=False

    auto_weight : bool
        Whether to automatically calculate class weights based on the training data

    manual_pos_weight : float or None
        Manual positive class weight to use if auto_weight=False

    reduction : str
        Specifies the reduction to apply to the output ('none', 'mean', 'sum')
    """

    def __init__(self, is_shorting=True, precision_factor=2.0, auto_weight=True,
                 manual_pos_weight=None, reduction='mean'):
        super(TradingBCELoss, self).__init__()
        self.is_shorting = is_shorting
        self.precision_factor = precision_factor
        self.auto_weight = auto_weight
        self.manual_pos_weight = manual_pos_weight
        self.reduction = reduction
        self.pos_weight = None  # Will be set in forward if auto_weight=True

        # Initialize the base BCE loss
        if not auto_weight and manual_pos_weight is not None:
            self.base_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([manual_pos_weight]),
                reduction='none'
            )
        else:
            self.base_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def calculate_class_weights(self, targets):
        """Calculate class weights based on the class distribution in the batch"""
        # Count positive and negative samples
        n_positives = torch.sum(targets).item()
        n_negatives = targets.numel() - n_positives

        # Avoid division by zero
        if n_positives == 0:
            n_positives = 1
        if n_negatives == 0:
            n_negatives = 1

        # Calculate positive weight (negative/positive ratio)
        pos_weight = n_negatives / n_positives

        # Print class balance information periodically (every 100 batches)
        # if np.random.random() < 0.01:  # Print info for approximately 1% of batches
        #     print(f"Class balance - Positives: {n_positives}, Negatives: {n_negatives}, Weight: {pos_weight:.4f}")

        return torch.tensor([pos_weight], device=targets.device)

    def forward(self, predictions, targets):
        """
        Forward pass for the custom loss function

        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions (logits)

        targets : torch.Tensor
            Binary target values (0 or 1)

        Returns:
        --------
        loss : torch.Tensor
            Calculated loss value
        """
        # Update pos_weight if using auto-weighting
        if self.auto_weight:
            self.pos_weight = self.calculate_class_weights(targets)
            # Apply pos_weight to the base criterion
            self.base_criterion.pos_weight = self.pos_weight

        # Calculate base BCE loss (element-wise, no reduction yet)
        base_loss = self.base_criterion(predictions, targets)

        # Apply shorting strategy modifications if is_shorting=False
        if not self.is_shorting:
            # Create masks for different error types
            # False positives: predicted positive (>0) but actually negative (0)
            false_positives = (torch.sigmoid(predictions) > 0.5) & (targets == 0)

            # False negatives: predicted negative (≤0) but actually positive (1)
            false_negatives = (torch.sigmoid(predictions) <= 0.5) & (targets == 1)

            # Increase weight for false positives (improve precision)
            modified_loss = base_loss.clone()
            modified_loss[false_positives] *= self.precision_factor

            # Decrease weight for false negatives (since we don't short)
            modified_loss[false_negatives] /= self.precision_factor

            # Use the modified loss
            loss = modified_loss
        else:
            # Use the base loss without modification for shorting strategy
            loss = base_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss