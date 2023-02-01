import torch
import torch.nn as nn

from tqdm import tqdm

from .utils import get_mind_iter

import torchmetrics

from datetime import datetime

import os

class Trainer(nn.Module):
    def __init__(
        self,
        time_dist_content,
        user_encoder,
        config
    ) -> None:
        super().__init__()
        self.config = config
        self.time_dist_content = time_dist_content
        self.user_encoder = user_encoder
        self.optimizer = torch.optim.Adam(
            list(self.time_dist_content.parameters()) + list(self.user_encoder.parameters()),
            lr=self.config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.auc = torchmetrics.AUROC(task="multiclass", num_classes=self.config.npratio + 1)

        self.train_iter = None
        self.val_iter = None
        self.init_data()

    def set_device(self, device):
        self.device = device
        self.to(device)
    

    def init_data(self, **kwargs):
        self.train_iter, self.val_iter = get_mind_iter(
            batch_size=self.config.batch_size,
            max_title_length=self.config.max_title_length,
            history_size=self.config.history_size,
            word_dict=self.time_dist_content.module.word2idx_dict,
            npratio=self.config.npratio, # negative positive ratio (4 = 4 negtive, 1 positive)
            MIND_type=self.config.MIND_type,
            **kwargs
        )
        

    
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
        encoded_hist_input = self.time_dist_content(hist_input)
        user_representations = self.user_encoder(encoded_hist_input)
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

    def fit(self, epochs=10):
        for epoch in range(0, epochs):
            epoch_loss, epoch_auc = self._fit_epoch(self.train_iter, epoch)
            eval_loss, eval_auc = self.eval()
            print("eval result loss:", eval_loss, "eval auc:", eval_auc)

    def _fit_epoch(self, data_iter, epoch):
        self.time_dist_content.train()
        self.user_encoder.train()
        losses_sum = 0
        auc_sum = 0

        tqdm_bar = tqdm(enumerate(data_iter), desc=f"Epoch {epoch}")
        for i, batch in tqdm_bar:
            loss, auc = self._fit_batch(batch)
            losses_sum += loss
            auc_sum += auc

            # update tqdm 
            current_mean_loss = losses_sum / (i + 1)
            tqdm_bar.set_description(f"Epoch {epoch} - mean_loss: {current_mean_loss:2.3f} - batch_loss: {loss:2.3f} - batch_auc: {auc:2.3f}")
        return losses_sum / (i + 1), auc_sum / (i + 1)

    def _fit_batch(self, batch):
        hist_input, candidate_input, labels = self.parse_batch(batch)
        self.optimizer.zero_grad()
        preds = self.forward(hist_input, candidate_input)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        auc = self.auc(preds, labels)
        return loss.item(), auc.item()

    def eval(self):
        self.time_dist_content.eval()
        self.user_encoder.eval()
        losses_sum = 0
        auc_sum = 0

        tqdm_bar = tqdm(enumerate(self.val_iter), desc=f"Evaluation")
        for i, batch in tqdm_bar:
            loss, auc = self._eval_batch(batch)
            losses_sum += loss
            auc_sum += auc

        return losses_sum / (i + 1), auc_sum / (i + 1)

    def _eval_batch(self, batch):
        hist_input, candidate_input, labels = self.parse_batch(batch)
        with torch.no_grad():
            preds = self.forward(hist_input, candidate_input)
            loss = self.criterion(preds, labels)
            auc = self.auc(preds, labels)
        return loss.item() , auc.item()

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

        user_history = torch.LongTensor(batch['clicked_title_batch'])
        candidate_titles = torch.LongTensor(batch['candidate_title_batch'])
        # convert one hot to index labels
        labels = torch.argmax(
            torch.FloatTensor(batch['labels']),
            dim=-1
        )

        # to torch device
        user_history = user_history.to(self.device)
        candidate_titles = candidate_titles.to(self.device)
        labels = labels.to(self.device)

        return(
            user_history,
            candidate_titles,
            labels
        )

    def save(self, dir):
        # add timestamp
        dir = os.path.join(dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(dir, exist_ok=True)

        trainer_path = os.path.join(dir, "trainer.pt")
        news_path = os.path.join(dir, "news_encoder.pt")
        user_path = os.path.join(dir, "user_encoder.pt")
        torch.save(
            {
                'config' : self.config,
                'optimizer' : self.optimizer.state_dict(),
            },
            trainer_path
        )
        self.time_dist_content.module.save(
            news_path
        )
        self.user_encoder.save(
            user_path
        )
        
