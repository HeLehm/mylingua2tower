import torch
import torch.nn as nn

from tqdm import tqdm


class Trainer():
    def __init__(
        self,
        time_dist_content,
        user_encoder
    ) -> None:
        self.time_dist_content = time_dist_content
        self.user_encoder = user_encoder
        self.optimizer = torch.optim.Adam(
            list(self.time_dist_content.parameters()) + list(self.user_encoder.parameters()),
            lr=0.001
        )
        self.criterion = nn.CrossEntropyLoss()

    
    def forward(self, hist_input, candidate_input):
        """
        Parameters
        ----------
        hist_input : torch.LongTensor
            Tensor of shape (batch_size, seq_count_in_batch, content_tower.max_seq_len) containing the contextualized nomalized text vectors.
        candidate_input : torch.LongTensor
            Tensor of shape (batch_size, seq_count_in_batch, content_tower.max_seq_len) containing the contextualized nomalized text vectors.
        Returns
        -------
        torch.FloatTensor
            Tensor of shape (batch_size, seq_count_in_batch, 1) containing the scores.
        """
        user_representations = self.user_encoder(hist_input)
        candidate_representations = self.time_dist_content(candidate_input)

        # reshape for dot product
        user_representations = user_representations.reshape(
            candidate_representations.shape[0],
            candidate_representations.shape[-1],
            1
        )
        
        # shape : batch_size, candidates_len
        candidate_cosine_sim = torch.matmul(
            candidate_representations,
            user_representations
        ).squeeze(-1) 

        #softmax over candidates
        preds = torch.softmax(candidate_cosine_sim, dim=-1)
        
        return preds

    def fit(self, data_iter, epochs=10):
        for epoch in range(0, epochs):
            epoch_loss = self._fit_epoch(data_iter, epoch)
            # eval


    def _fit_epoch(self, data_iter, epoch):
        self.time_dist_content.train()
        self.user_encoder.train()
        losses_sum = 0

        tqdm_bar = tqdm(enumerate(data_iter), desc=f"Epoch {epoch}")
        for i, batch in tqdm_bar:
            loss = self._fit_batch(batch)
            losses_sum += loss
            # update tqdm 
            current_mean_loss = losses_sum / (i + 1)
            tqdm_bar.set_description(f"Epoch {epoch} - mean_loss: {current_mean_loss:2.3f} - batch_loss: {loss:2.3f}")
        return losses_sum / (i + 1)

    def _fit_batch(self, batch):
        hist_input, candidate_input, labels = self.parse_batch(batch)
        self.optimizer.zero_grad()
        preds = self.forward(hist_input, candidate_input)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def parse_batch(self, batch):
        """
        Return
        ------
        
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
        user_history : torch.LongTensor
            Tensor of shape (batch_size, history_len, title_length)
        candidate_titles : torch.LongTensor
            candidate_len = npratio + 1
            Tensor of shape (batch_size, candidates_len, title_length)
        labels : torch.LongTensor
            Tensor of shape (batch_size)
        """
        # convert one hot to index labels

        return(
            torch.LongTensor(batch['clicked_title_batch']),
            torch.LongTensor(batch['candidate_title_batch']),
            torch.FloatTensor(batch['labels'])
        )