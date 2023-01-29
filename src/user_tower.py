import torch
import torch.nn as nn


class DummyUserTower(nn.Module):
    """
    User tower of the model.
    Dummy implementation, just returns the mean of the content tower outputs.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, content_tower_outputs):
        """
        Parameters
        ----------
        content_tower_outputs : torch.FloatTensor
            Tensor of shape (batch_size, texts in batch, hidden_size) containing the contextualized nomalized text vectors.
        """
        mean =  torch.mean(content_tower_outputs, dim=1)
        # l2 normalize
        mean = mean / torch.norm(mean, dim=1, keepdim=True)
        return mean

    
