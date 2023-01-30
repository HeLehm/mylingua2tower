from typing import List

import torch
from tqdm import tqdm
import torch.nn as nn

from .content_tower import MHAContentTower
from .user_tower import DummyUserTower
from .time_distributed import TimeDistributed
from .tokenize import tokenize


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self._build_graph()
        
    def _build_contentencoder(self):
        self.content_tower = MHAContentTower(self.config)
        self.time_dist_content_tower = TimeDistributed(self.content_tower)

    def _build_userencoder(self):
        if self.time_dist_content_tower is None:
            raise ValueError("Content tower must be built before user tower")
        self.user_tower = DummyUserTower(self.time_dist_content_tower)

    def _build_graph(self):
        self._build_contentencoder()
        self._build_userencoder()

    def vectorize_texts(self, texts: List[List[str]]):
        """
        Parameters
        ----------
        texts : List[List[str]]
            List of list of strings. Each string is a text.
        Returns
        -------
        torch.LongTensor
            Tensor of shape (batch_size, seq_count_in_batch, content_tower.max_seq_len) containing the contextualized nomalized text vectors.
        """

        tokenized_texts = texts.copy()
        for i, te in enumerate(texts):
            for j, t in enumerate(te):
                tokenized_texts[i][j] = tokenize(t)
        
        tensors = []
        for tokenized_text in tokenized_texts:
            tensor = self.content_tower.texts_to_ids(tokenized_text)
            tensors.append(tensor)
        
        # pad to match max_candidates_len
        max_candidate_len = max([t.shape[0] for t in tensors])
        for i, tensor in enumerate(tensors):
            tensors[i] = torch.cat([tensor, torch.zeros((max_candidate_len - tensor.shape[0], tensor.shape[1]), dtype=torch.long)])
        
        return torch.stack(tensors)

    def forward(self, hist_input, candidate_input):
        """
        Parameters
        ----------
        hist_input : torch.LongTensor
            Tensor of shape (batch_size, history_size, seq_len) containing the contextualized nomalized text vectors.
        candidate_input : torch.LongTensor
            Tensor of shape (batch_size, candidates_len, seq_len) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, candidates_len) containing the softmax probs of candidates.
        """
        user_representations = self.user_tower(hist_input) # shape : batch_size, embedding_dim
        candidate_representations = self.time_dist_content_tower(candidate_input) # shape : batch_size, candidates_len, embedding_dim

        #dot products
        # TODO check correctness
        candidate_cosine_sim = torch.matmul(candidate_representations, user_representations.reshape(candidate_representations.shape[0], candidate_representations.shape[-1], 1) ) # shape : batch_size, candidates_len, 1
        candidate_cosine_sim = candidate_cosine_sim.squeeze(-1)

        # nomalize to 0-1
        preds = torch.sigmoid(candidate_cosine_sim)

        return preds


class Trainer():
    def __init__(self, model):
        self.model : Model = model
        self.config = model.config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss() # multi-label 
        
    def fit(self, epochs, train_loader, valid_loader):
        for epoch in range(epochs):
            loss = self.train(train_loader)
            print(f"Epoch {epoch} loss: {loss}")
            with torch.no_grad():
                loss = self.validate(valid_loader)
                print(f"Epoch {epoch} validation loss: {loss}")
                
            self.validate(valid_loader)
    
    def train(self, train_loader):
        self.model.train()
        train_loss = 0.
        for batch in tqdm(train_loader):
            hist_texts, candidate_texts, labels = batch 
            
            hist_input = self.model.vectorize_texts(hist_texts) # shape: batch_size, history_size, seq_len
            candidate_input = self.model.vectorize_texts(candidate_texts) # shape: batch_size, candidates_len, seq_len
            
            self.optimizer.zero_grad()
            preds = self.model(hist_input, candidate_input)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss

    def validate(self, valid_loader):
        self.model.eval()
        valid_loss = 0.
        for batch in tqdm(valid_loader):
            hist_texts, candidate_texts, labels = batch 
            
            hist_input = self.model.vectorize_texts(hist_texts)
            candidate_input = self.model.vectorize_texts(candidate_texts)

            preds = self.model(hist_input, candidate_input)
            loss = self.criterion(preds, labels)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    


            


