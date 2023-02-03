from recommenders.models.newsrec.models.nrms import NRMSModel as _NRMSModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention

import tensorflow.keras as keras
from tensorflow.keras import layers

import os
import json
import pickle
import numpy as np

from typing import List

class NRMSModel(_NRMSModel):
    
    @classmethod
    def load_encoders(cls, model_dir, **kwargs):
        # get paths
        checkpoint_path, _, word_2_idx_path = cls._get_paths(model_dir)

        # load encoders
        word_2_idx_path = os.path.join(model_dir, "nrms_word_2_idx.pkl")
        with open(word_2_idx_path, "rb") as f:
            word_2_idx = pickle.load(f)

        # load model
        model = keras.models.load_model(checkpoint_path)

        # build encoders
        userencoder_model = cls._build_userencoder_from_pretrained(model)
        newsencoder_model = cls._build_newsencoder_from_pretrained(model)

        return NewsEncoder(newsencoder_model, word_2_idx, **kwargs), UserEncoder(userencoder_model)

    @classmethod
    def _build_userencoder_from_pretrained(cls, model):
        u_s_att = model.get_layer('user_encoder').get_layer('self_attention_1')
        u_input = layers.Input(shape=(u_s_att.input[0].shape[1],u_s_att.input[0].shape[2]), name='clicked_title_batch')
        u_s_att_out = u_s_att([u_input, u_input, u_input])
        u_att_out = model.get_layer('user_encoder').get_layer('att_layer2_1')(u_s_att_out)
        return keras.Model(u_input, u_att_out, name='user_encoder')

    @classmethod
    def _build_newsencoder_from_pretrained(cls, model):
        t = model.get_layer('time_distributed_1').layer
        return keras.Model(t.input, t.output, name='news_encoder')

    @classmethod 
    def _get_paths(cls, model_dir):
        checkpoint_path = os.path.join(model_dir, "nrms_ckpt")
        hparams_path = os.path.join(model_dir, "nrms_hparams.json")
        word_2_idx_path = os.path.join(model_dir, "nrms_word_2_idx.pkl")
        return (
            checkpoint_path,
            hparams_path,
            word_2_idx_path
        )


    def save(self, model_dir):
        # get paths
        checkpoint_path,\
            hparams_path,\
            word_2_idx_path = self._get_paths(model_dir)

        # save weights
        self.model.save(checkpoint_path)
        # save hparams
        with open(hparams_path, "w") as f:
            h_json = self.hparams._values
            json_str = json.dumps(h_json)
            f.write(json_str)
            
        # save word_2_idx
        with open(word_2_idx_path, "wb") as f:
            pickle.dump(self.train_iterator.word_dict, f)


    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _get_opt(self):
        """
        Override lr -> learning_rate
        Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(learning_rate=lr)

        return train_opt


class NewsEncoder():
    
    def __init__(self, model, word2idx):
        self.model = model
        self.word2idx = word2idx
    
    def tokenize(self, news_titles : List[str]):
        """
        NOTE: Code Duplication from MINDIterator
        """
        title_size = self.model.input.shape[1]
        print("title sizeddd ",title_size)
        news_title_index = np.zeros(
            (len(news_titles), title_size), dtype="int32"
        )

        for news_index in range(len(news_titles)):
            title = news_titles[news_index]
            for word_index in range(min(title_size, len(title))):
                if title[word_index].lower() in self.word2idx:
                    news_title_index[news_index, word_index] = self.word2idx[
                        title[word_index].lower()
                    ]
                elif 'unk' in self.word2idx:
                    news_title_index[news_index, word_index] = self.word2idx['unk']
        
        return news_title_index

    def predict(self, tokenized_texts):
        return self.model.predict(tokenized_texts)

class UserEncoder():
    
    def __init__(self, model):
        self.model = model

    def pad_crop_input(self, single_user_encoded_news):
        """
        Parameters
        ----------
        encoded_news : np.array
            shape (None, news_embedding_size)
        Returns
        -------
        np.array
            shape (1, his_size, news_embedding_size)
        """
        his_size = self.model.input.shape[1]
        news_embedding_size = self.model.input.shape[2]

        # cropped and padded news
        cp_news =  np.zeros((1, his_size, news_embedding_size), dtype="float32")

        if single_user_encoded_news.size == 0:
            return cp_news

        # TODO: test extensively
        cp_news[0, -single_user_encoded_news.shape[0]:, :] = single_user_encoded_news[-his_size:, :]

        return cp_news


    def predict(self, encoded_news):
        return self.model.predict(encoded_news)