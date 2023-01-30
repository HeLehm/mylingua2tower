import os
from typing import Literal
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.models.newsrec.newsrec_utils import prepare_hparams

import torch


from recommenders.models.newsrec.models.nrms import NRMSModel

from recommenders.models.newsrec.io.mind_iterator import MINDIterator

from .paths import get_mind_dir


def setup_mind(
    MIND_type : Literal['demo', 'small', 'large'] = 'demo', dir=None,
    force_download=False,
    remove_embeddings=True,
    **kwargs
):
    """
    Setup MIND dataset.
    Args:
        MIND_type (str): MIND type, one of ['demo', 'small', 'large']
        root_path (str): root path of the dataset
    """
    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)
    if dir is None:
        dir = os.path.join(get_mind_dir(), MIND_type)
    os.makedirs(dir, exist_ok=True)
    
    train_news_path = os.path.join(dir, 'train' , r'news.tsv')
    train_behaviors_path = os.path.join(dir, 'train' , r'behaviors.tsv')
    dev_news_path = os.path.join(dir, 'valid' , r'news.tsv')
    dev_behaviors_path = os.path.join(dir, 'valid' , r'behaviors.tsv')

    wordEmb_file = os.path.join(dir, "utils", "embedding.npy")
    userDict_file = os.path.join(dir, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(dir, "utils", "word_dict.pkl")
    yaml_file = os.path.join(dir, "utils", r'nrms.yaml')

    if force_download or not os.path.exists(train_news_path):
        download_deeprec_resources(mind_url, os.path.join(dir, 'train'), mind_train_dataset)
    
    if force_download or not os.path.exists(dev_news_path):
        download_deeprec_resources(mind_url, os.path.join(dir, 'valid'), mind_dev_dataset)

    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               os.path.join(dir, 'utils'), mind_utils)

    
    """
    if remove_embeddings:
        if os.path.exists(os.path.join(dir, 'train', 'entity_embedding.vec')):
            os.remove(os.path.join(dir, 'train', 'entity_embedding.vec'))
        if os.path.exists(os.path.join(dir, 'train', 'relation_embedding.vec')):
            os.remove(os.path.join(dir, 'train', 'relation_embedding.vec'))
        if os.path.exists(os.path.join(dir, 'valid', 'entity_embedding.vec')):
            os.remove(os.path.join(dir, 'valid', 'entity_embedding.vec'))
        if os.path.exists(os.path.join(dir, 'valid', 'relation_embedding.vec')):
            os.remove(os.path.join(dir, 'valid', 'relation_embedding.vec'))
    """

    
    return train_news_path, \
            train_behaviors_path,\
            dev_news_path,\
            dev_behaviors_path


class MINDDataset(Dataset):
    def _parse_tsv(self, path):
        with open(path, 'r') as f:
            for line in f:
                yield line.strip().split('\t')
    
class MINDInteractionDataset(MINDDataset):
    def __init__(self, behaviour_path) -> None:
        super().__init__()
        self.behaviour_data = list(self._parse_behaviour(behaviour_path))
    
    def _parse_behaviour(self, behaviour_path):
        for line in self._parse_tsv(behaviour_path):
            id, user_id, time, history, impressions = line
            yield id, user_id, time, self._parse_history(history), self._parse_impressions(impressions)

    def _parse_impressions(self, impressions):
        for impression in impressions.split():
            news_id,label = impression.split('-')
            yield news_id, int(label)
    
    def _parse_history(self, history):
        for news_id in history.split():
            yield news_id

    def __len__(self):
        return len(self.behaviour_data)

    def __getitem__(self, index):
        return self.behaviour_data[index]

class MINDNewsDataset(MINDDataset):
    def __init__(self, news_path) -> None:
        super().__init__()
        self.news_data = {
            news_id: (category, subcategory, title, abstract, url, title_entities, abstract_entities)
            for news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities in self._parse_news(news_path)
        }
    
    def _parse_news(self, news_path):
        for line in self._parse_tsv(news_path):
            news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = line
            yield news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities

    def __len__(self):
        return len(self.news_data)
    
    def __getitem__(self, news_id):
        return self.news_data[news_id]


class MINDDataLoader():
    def __init__(self, batch_size, **kwargs) -> None:
        self.batch_size = batch_size
        self.news = None
        self.interactions = None
        self.load_data(**kwargs)
        
    def load_data(self,validation=False, **kwargs):
        train_news_path, \
            train_behaviors_path,\
            dev_news_path,\
            dev_behaviors_path = setup_mind(**kwargs)
        
        if validation:
            news_path = dev_news_path
            behaviour_path = dev_behaviors_path
        else:
            news_path = train_news_path
            behaviour_path = train_behaviors_path
        
        self.news = MINDNewsDataset(news_path)
        self.interactions = MINDInteractionDataset(behaviour_path)

    def __iter__(self):
        for i in range(0, len(self.interactions), self.batch_size):
            batch = self.interactions[i:i+self.batch_size]
            yield self._collate(batch)

    def _collate(self, batch):
        batch_hist_texts = []
        batch_candidate_texts = []
        batch_labels = []

        for id, user_id, time, history, impressions in batch:
            impression_labels = []
            hist_texts = []
            for news_id in history:
                hist_texts.append(self.news[news_id][2])
            batch_hist_texts.append(hist_texts)

            candidate_texts = []
            for i, (news_id, label) in enumerate(impressions):
                
                candidate_texts.append(self.news[news_id][2])
                if label == 1:
                    impression_labels.append(i)

            assert len(impression_labels), "no positive impression in id:{}".format(id)
            batch_candidate_texts.append(candidate_texts)
            batch_labels.append(impression_labels)

        assert len(batch_hist_texts) == len(batch_candidate_texts) == len(batch_labels), "batch size must match, but got {} {} {}".format(len(batch_hist_texts), len(batch_candidate_texts), len(batch_labels))


        max_candidates_len = max([len(candidates) for candidates in batch_candidate_texts])
        label_tensor = torch.zeros((len(batch_labels), max_candidates_len), dtype=torch.float)
        for i, labels in enumerate(batch_labels):
            for label in labels:
                label_tensor[i][label] = 1
        
        return (
            batch_hist_texts, # List[List[str]] shape: [batch_size, history_size]
            batch_candidate_texts, # List[List[str]] shape: [batch_size, candidate_size]
            label_tensor # Tensor[int] shape: [batch_size, candidate_size]
        )
    
    def __len__(self):
        return len(self.interactions) // self.batch_size