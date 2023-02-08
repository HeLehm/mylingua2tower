import tensorflow as tf
import numpy as np

from recommenders.models.newsrec.newsrec_utils import word_tokenize
from recommenders.models.newsrec.io.mind_iterator import MINDIterator as _MINDIterator

class MINDIterator(_MINDIterator):

    def init_news(self, news_file):
        """
        Override: add unk token if possible
        init news information given news file, such as news_title_index and nid2index.
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
                title = word_tokenize(title)
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
                elif 'unk' in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict['unk']