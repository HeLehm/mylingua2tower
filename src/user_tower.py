import torch
import torch.nn as nn

from .utils.fastformer import FastformerEncoder

from torch.nn.functional import normalize
from transformers import BertConfig

class UserTower(nn.Module):
    """
    User tower of the model.
    Dummy implementation, just returns the mean of the content tower outputs.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()

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
            Tensor of shape (batch_size, history_size, hidden_size) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, hidden_size) containing the user vectors.
        """
        user_vec = self._forward_history(x)
        user_vec = normalize(user_vec, p=2.0, dim=-1)
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
        return torch.mean(content_tower_outputs, dim=1)
       

class MHAUserTower(UserTower):
    #TODO: think about possitional embeddings

    def __init__(self, config : BertConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = FastformerEncoder(config)
    
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
        # use multi-head attention to get the user vector

        # mask = zero where content_tower_outputs is zero
        #TODO: check if this is correct
        mask = torch.sum(content_tower_outputs, dim=-1).bool().float() # (batch_size, history_size)

        # attention
        user_vec = self.attention(content_tower_outputs, mask)

        return user_vec
