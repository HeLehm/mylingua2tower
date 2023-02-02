from typing import Literal

from .utils import setup_mind
from ..word_embedding import create_torch_embedding_layer
from .mind_iterator import MINDIterator

def get_mind_iter(
    batch_size: int,
    max_title_length: int,
    history_size: int,
    word_dict = None,
    npratio : int = 4,
    MIND_type: Literal['demo', 'small', 'large'] = 'demo',
    mind_data_dir=None,
    **kwargs,
):  
    """
    Return
    ------
        iterator: MINDIterator object
        train_news_path: path to train news file
        train_behaviors_path: path to train behaviors file
        dev_news_path: path to dev news file
        dev_behaviors_path: path to dev behaviors file
    """
    if word_dict is None:
        _, word_dict = create_torch_embedding_layer(**kwargs)


    tnp, tbp, dnp, dbp, udf = setup_mind(MIND_type, mind_data_dir, **kwargs)

    mind_kwargs = {
        'batch_size': batch_size,
        'max_title_length': max_title_length,
        'history_size': history_size,
        'wordDict': word_dict,
        'userDict_file': udf,
        'npratio': npratio,
        **kwargs
    }

    train_iterator = MINDIterator(tnp, tbp, **mind_kwargs)
    dev_iterator = MINDIterator(dnp, dbp, **mind_kwargs)

    return train_iterator, dev_iterator