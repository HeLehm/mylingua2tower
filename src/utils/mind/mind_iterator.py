# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Union, Dict
import tensorflow as tf
import numpy as np

from ..tokenize_def import tokenize

from recommenders.models.newsrec.io.mind_iterator import MINDIterator as _MINDIterator



class MINDIterator(_MINDIterator):
    """
    MIND dataset iterator
    Changes to the original MINDIterator:
    (init)
    - Added support for dict instead of just file for wordDict
    - params ddefined seperately inseatd of hparam object
    (init_news)
    - Added 'unk' (unknown) token 
    - Changed to use (nltk)tokenize instead of theirs
    """
    def __init__(
        self,
        news_file : str,
        behaviors_file : str,
        batch_size : int,
        max_title_length : int,
        history_size : int,
        wordDict : Union[str, Dict],
        userDict_file : str,
        npratio=-1,
        col_spliter="\t",
        ID_spliter="%",
        **kwargs,
    ):
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = batch_size
        self.title_size = max_title_length
        self.his_size = history_size
        self.npratio = npratio

        if isinstance(wordDict, dict):
            self.word_dict = wordDict
        else:
            self.word_dict = self.load_dict(wordDict)

        self.uid2index = self.load_dict(userDict_file)

    def __iter__(self):
        return self.load_data_from_file(self.news_file, self.behaviors_file)

    def __len__(self):
        return len(self.labels) + 1 // self.batch_size
        
    def init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.nid2index = {}
        news_title = [""]

        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = tokenize(title)
                news_title.append(title)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index].lower() in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]
                else:
                    self.news_title_index[news_index, word_index] = self.word_dict["unk"]