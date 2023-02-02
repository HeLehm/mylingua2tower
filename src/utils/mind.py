import os
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set as _get_mind_data_set
from recommenders.models.newsrec.newsrec_utils import prepare_hparams

from .paths import get_mind_dir
from .glove import dowload_glove, parse_and_save_glove_array

def get_mind_train(MIND_type, mind_data_dir=None, force_download=False):
    mind_url, mind_train_dataset, _, _ = _get_mind_data_set(MIND_type)
    mind_data_dir = _get_mind_dir(MIND_type, mind_data_dir)
    train_dir = os.path.join(mind_data_dir, 'train')

    train_news_path = os.path.join(train_dir, r'news.tsv')
    train_behaviors_path = os.path.join(train_dir , r'behaviors.tsv')

    if force_download or not os.path.exists(train_news_path) or not os.path.exists(train_behaviors_path):
        download_deeprec_resources(mind_url, train_dir, mind_train_dataset)

    return train_news_path, train_behaviors_path

def get_mind_val(MIND_type, mind_data_dir=None, force_download=False):
    mind_url, _, mind_dev_dataset, _ = _get_mind_data_set(MIND_type)
    mind_data_dir = _get_mind_dir(MIND_type, mind_data_dir)
    dev_dir = os.path.join(mind_data_dir, 'valid')

    dev_news_path = os.path.join(dev_dir, r'news.tsv')
    dev_behaviors_path = os.path.join(dev_dir , r'behaviors.tsv')

    if force_download or not os.path.exists(dev_news_path) or not os.path.exists(dev_behaviors_path):
        download_deeprec_resources(mind_url, dev_dir, mind_dev_dataset)

    return dev_news_path, dev_behaviors_path

def get_mind_utils(MIND_type, mind_data_dir=None, force_download=False):
    """
    Get the utils files for MIND dataset
    Return
    ------
    wordEmb_file: str
        the path of word embedding file
    userDict_file: str
        the path of user dictionary file
    wordDict_file: str
        the path of word dictionary file
    yaml_file: str
        the path of nrms yaml file
    """
    _, _, _, mind_utils = _get_mind_data_set(MIND_type)
    mind_data_dir = _get_mind_dir(MIND_type, mind_data_dir)
    utils_dir = os.path.join(mind_data_dir, 'utils')

    wordEmb_file = os.path.join(utils_dir, "embedding.npy")
    userDict_file = os.path.join(utils_dir, "uid2index.pkl")
    wordDict_file = os.path.join(utils_dir, "word_dict.pkl")
    yaml_file = os.path.join(utils_dir, r'nrms.yaml')

    if force_download or not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               utils_dir, mind_utils)

    return wordEmb_file, userDict_file, wordDict_file, yaml_file


def _get_mind_dir(MIND_type, mind_data_dir=None):
    if mind_data_dir is None:
        mind_data_dir = os.path.join(get_mind_dir(), MIND_type)
    os.makedirs(mind_data_dir, exist_ok=True)
    return mind_data_dir


def get_hprarams(
    MIND_type,
    mind_data_dir=None,
    force_download=False,
    glove_name='glove.6B',
    word_emb_dim=300,
    **kwargs,
):
    """
    Get the hyper-parameters for MIND dataset
    Return
    ------
    hparams: dict
        the hyper-parameters for MIND dataset
    """
    assert word_emb_dim in [50, 100, 200, 300], "word_emb_dim should be in [50, 100, 200, 300]"
    _, userDict_file, _, yaml_file = get_mind_utils(
        MIND_type, mind_data_dir, force_download
    )

    # GloVe setup
    glove_name_d = f'{glove_name}.{word_emb_dim}d'
    dowload_glove(glove_name, **kwargs)
    wordDict_file, wordEmb_file = parse_and_save_glove_array(glove_name_d=glove_name_d, padding=True, **kwargs)
    
    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file, 
        userDict_file=userDict_file,
        glove_name=glove_name,
        **kwargs
    )
    return hparams



