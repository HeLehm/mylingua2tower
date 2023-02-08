import os
import sys

path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(path_to_add)


import unittest
from tempfile import TemporaryDirectory
import numpy as np

from mylingua2tower.utils.mind import get_hprarams
from mylingua2tower.iterator import MINDIterator
from mylingua2tower.nrms import NRMSModel


MIND_type = 'demo'
seed = 42

class TestLoad(unittest.TestCase):
    def setUp(self) -> None:
        hparams = get_hprarams(MIND_type, his_size=5) # his_size=5 is a hack to make the test run faster
        iterator = MINDIterator
        self.model = NRMSModel(hparams, iterator, seed=seed)
        return super().setUp()
        
    def test(self):
        with TemporaryDirectory() as tempdir_name:
            self.model.save(tempdir_name)
            newsencoder, userencoder = NRMSModel.load_encoders(tempdir_name)
            
            example_news_titles = [
                "The Latest: Trump says he's 'not happy' with Fed",
                "The 10 Best Movies of 2019",
                "The 10 Best TV Shows of 2019",
                "Brexit: Boris Johnson's deal with the EU explained",
                "Breakingviews - Brexit: Boris Johnson's deal with the EU explained",
            ]

            tokenized_news_titles = newsencoder.tokenize(example_news_titles)

            news_embeddings = newsencoder.model.predict(tokenized_news_titles)

            padded_user = userencoder.pad_crop_input(news_embeddings) # user has seen all news
            user_embeddings = userencoder.predict(padded_user)

            preds = np.dot(news_embeddings, user_embeddings.T)
            # softmax preds 
            preds = np.exp(preds) / np.sum(np.exp(preds), axis=0)

            old_preds = self.model.model.predict([
                np.expand_dims(tokenized_news_titles, axis=0),
                np.expand_dims(tokenized_news_titles, axis=0)
            ])

            #sqeeze
            old_preds = np.squeeze(old_preds)
            preds = np.squeeze(preds)

            for i in range(preds.shape[0]):
                    self.assertAlmostEqual(preds[i], old_preds[i], places=3)


if __name__ == '__main__':
    unittest.main()