from typing import List
import torch
import torch.nn as nn
from transformers import BertConfig

from .word_embedding import create_embedding_layer
from .fastformer import FastformerEncoder


class MHAContentTower(nn.Module):
    """
    Content tower of the model.
    Uses a multi headed FastformerEncoder to create contextualized normalized text vectors.
    """
    def __init__(self, config : BertConfig):
        super().__init__()
        self.fastformer_model = FastformerEncoder(config)
        self.word_embedding, self.word2idx_dict  = create_embedding_layer(glove_name=config.glove_name, dim=config.word_embedding_dim)
        self.max_seq_len = config.max_seq_len
        
    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids : torch.LongTensor
            Tensor of shape (batch_size, seq_len) containing the indices of the words.
        Returns
        -------
        text_vec : torch.FloatTensor
            Tensor of shape (batch_size, hidden_size) containing the contextualized nomalized text vectors.
        """
        mask = input_ids.bool().float()
        embds = self.word_embedding(input_ids)
        text_vec = self.fastformer_model(embds,mask)
        # euclidian normlization
        text_vec = text_vec / torch.norm(text_vec, dim=-1, keepdim=True)
        return text_vec

    def word2idx(self, word : str) -> int:
        """
        Parameters
        ----------
        word : str
            Word to get the index of.
        Returns
        -------
        idx : int
            Index of the word according to embeddinglayer.
        """
        # plus 1 because of padding
        if word not in self.word2idx_dict:
            return self.word2idx_dict['unk'] + 1
        return self.word2idx_dict[word] + 1
    
    def texts_to_ids(self, tokenized_texts : List[List[str]]) -> torch.LongTensor:
        """
        Parameters
        ----------
        tokenized_texts : list of list of str
            List of tokenized texts.
        Returns
        -------
        idxs : torch.LongTensor
            Tensor of shape (batch_size, seq_len) containing the indices of the words.
        """
        # get word indices
        word_idxs = [[self.word2idx(word) for word in text] for text in tokenized_texts]
        # pad or trim to max seq length
        word_idxs = [text[:self.max_seq_len] + [0] * (self.max_seq_len - len(text)) for text in word_idxs]
        # turn to tensor
        return torch.LongTensor(word_idxs)