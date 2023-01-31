import os
from typing import Literal
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set 


from ..paths import get_mind_dir


def setup_mind(
    MIND_type : Literal['demo', 'small', 'large'] = 'demo',
    dir=None,
    force_download=False,
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
            dev_behaviors_path,\
            userDict_file