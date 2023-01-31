import torch
import torch.nn as nn

from .utils.time_distributed import TimeDistributed

from torch.nn.functional import normalize


class UserTower(nn.Module):
    """
    User tower of the model.
    Dummy implementation, just returns the mean of the content tower outputs.
    """
    def __init__(self, time_distributed_content_tower) -> None:
        super().__init__()
        self.time_distributed_content_tower = time_distributed_content_tower

    def _forward_history(self, content_tower_outputs):
        """
        Parameters
        ----------
        content_tower_outputs : torch.FloatTensor
            Tensor of shape (batch_size, history_size, hidden_size) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, hidden_size) containing the user vectors.
        """
        # will be implemented in child classes
        pass

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.LongTensor
            Tensor of shape (batch_size, history_size, seq_len) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, hidden_size) containing the user vectors.
        """
        click_title_presents = self.time_distributed_content_tower(x) # (batch_size, history_size, hidden_size)
        user_vec = self._forward_history(click_title_presents)
        return user_vec
        

        
class DummyUserTower(UserTower):
    def _forward_history(self, content_tower_outputs):
        """
        Parameters
        ----------
        content_tower_outputs : torch.FloatTensor
            Tensor of shape (batch_size, history_size, hidden_size) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, hidden_size) containing the user vectors.
        """
        # just return the mean of the content tower outputs
        mean = torch.mean(content_tower_outputs, dim=1)
        return normalize(mean, p=2.0, dim=-1)
       

    
