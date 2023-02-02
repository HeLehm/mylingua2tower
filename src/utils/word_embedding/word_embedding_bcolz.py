from .word_embedding_util import _create_torch_embedding_layer
from ..paths import get_bcolz_glove_files, get_glove_dir

import os
import bcolz
import numpy as np

import pickle

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

    # declare paths
    glove_path, glove_words_path, glove_idx_path, glove_vectors_path = \
        get_bcolz_glove_files(dir,glove_name_d)
    #check if files already exist

    if not force_redo \
        and os.path.exists(glove_path) \
        and os.path.exists(glove_words_path) \
        and os.path.exists(glove_idx_path) \
        and os.path.exists(glove_vectors_path):
        print('GloVe files already exist. Use force_redo=True to overwrite.')
        return
    
    # parse glove file
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{dir}/{glove_name_d}.dat', mode='w')

    with open(f'{dir}/{glove_name_d}.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir=f'{dir}/{glove_name_d}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{dir}/{glove_name_d}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{dir}/{glove_name_d}_idx.pkl', 'wb'))

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
        get_bcolz_glove_files(dir,glove_name_d)
    
    #load files
    #words = pickle.load(open(glove_words_path, 'rb'))
    word2idx = pickle.load(open(glove_idx_path, 'rb'))
    vectors = bcolz.open(glove_vectors_path)[:]

    # the order should be correct as created in parse_and_save_glove

    return word2idx, vectors # aka idx2emb

def create_torch_embedding_layer(**kwargs):
    return _create_torch_embedding_layer(**kwargs, load_glove=load_glove, parse_and_save_glove=parse_and_save_glove)