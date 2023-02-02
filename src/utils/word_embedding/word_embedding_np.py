import numpy as np
import pickle

from .word_embedding_util import _create_torch_embedding_layer

from ..paths import get_glove_dir , get_np_glove_files

def parse_and_save_glove(
    dir = get_glove_dir(),
    glove_name_d='glove.6B.50d',
    force_redo=False
):
    """
    Parse GloVe word embeddings and save them in a pickle file.
    Parameters
    ----------
    dir : str
        Path to the directory where the GloVe files are stored.
    glove_name_d : str
        Name of the GloVe file.
    force_redo : bool
        If True, overwrite existing files.
        If False, use existing files if they exist.
    """
    words = []
    idx = 0
    word2idx = {}
    vectors = []
    with open(f'{dir}/{glove_name_d}.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = np.array(vectors)
    # TODO: reshape?

    _, glove_words_path, glove_idx_path, glove_vectors_path = \
        get_np_glove_files(dir, glove_name_d)

    #save files
    pickle.dump(words, open(glove_words_path, 'wb'))
    pickle.dump(word2idx, open(glove_idx_path, 'wb'))
    np.save(glove_vectors_path, vectors)

def load_glove(
    dir = get_glove_dir(),
    glove_name_d='glove.6B.50d',
):
    """
    Load GloVe word embeddings.
    Returns
    -------
    word2idx : dict
        Dictionary moving from words to indices.
    vectors : bcolz.carray
        Array of GloVe word embeddings.
    """
    # get paths
    _, glove_words_path, glove_idx_path, glove_vectors_path = \
        get_np_glove_files(dir, glove_name_d)
    
    #load files
    #words = pickle.load(open(glove_words_path, 'rb'))
    word2idx = pickle.load(open(glove_idx_path, 'rb'))
    vectors = np.load(glove_vectors_path)

    # the order should be correct as created in parse_and_save_glove

    return word2idx, vectors

def create_torch_embedding_layer(**kwargs):
    return _create_torch_embedding_layer(load_glove=load_glove, parse_and_save_glove=parse_and_save_glove, **kwargs)
