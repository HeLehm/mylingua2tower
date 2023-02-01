from typing import Literal

from .utils import setup_mind
from .mind_iterator import MINDIterator

def get_mind_iter(
    batch_size: int,
    max_title_length: int,
    history_size: int,
    word_dict,
    npratio : int = 4,
    MIND_type: Literal['demo', 'small', 'large'] = 'demo',
    dir=None,
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
    tnp, tbp, dnp, dbp, udf = setup_mind(MIND_type, dir, **kwargs)

    mind_kwargs = {
        'batch_size': batch_size,
        'max_title_length': max_title_length,
        'history_size': history_size,
        'wordDict': word_dict,
        'userDict_file': udf,
        'npratio': npratio,
        **kwargs
    }

    train_iterator = MINDIterator(**mind_kwargs)
    t = train_iterator.load_data_from_file(tnp, tbp)

    dev_iterator = MINDIterator(**mind_kwargs)
    d = dev_iterator.load_data_from_file(dnp, dbp)

    return t, d
