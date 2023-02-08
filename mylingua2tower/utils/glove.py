import os
import io
import pickle
import numpy as np
import zipfile
import requests

from glob import glob

from .paths import get_glove_dir, get_glove_files

def download_glove(
    glove_name = 'glove.6B',
    glove_dir=None,
    force_download=False,
    **kwargs
):
    """
    Download GloVe word embeddings.
    Parameters
    ----------
    glove_name : str
        Name of the GloVe. e.g. 'glove.6B'
    glove_dir : str
        Path to the directory where the GloVe files are stored.
        default: None->get_glove_dir()
    force_download : bool
        If True, overwrite existing files.
    """
    if glove_dir is None:
        glove_dir = get_glove_dir()

    download_url = f'https://nlp.stanford.edu/data/{glove_name}.zip'

    #check if file already exists   
    path_wildcard = os.path.join(glove_dir, f'{glove_name}.*d.txt')
    if not force_download and len(glob(path_wildcard)) > 0:
        return
    print('Downloading GloVe word embeddings...')

    r = requests.get(download_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(glove_dir)
    z.close()


def parse_and_save_glove_array(
    glove_dir = get_glove_dir(),
    glove_name_d='glove.6B.50d',
    padding=True,
    force_redo=False,
    **kwargs
):
    """
    Parse GloVe word embeddings and save them in a pickle file.
    Parameters
    ----------
    glove_dir : str
        Path to the directory where the GloVe files are stored.
    glove_name_d : str
        Name of the GloVe file.
    force_redo : bool
        If True, overwrite existing files.
        If False, use existing files if they exist.
    """
    # get paths
    _, glove_words_path, glove_idx_path, glove_vectors_path = \
        get_glove_files(glove_dir, glove_name_d, padded=padding)

    if (
        not force_redo 
        and os.path.exists(glove_words_path)
        and os.path.exists(glove_idx_path)
        and os.path.exists(glove_vectors_path)
    ):
        return glove_idx_path, glove_vectors_path

    words = []
    idx = 0
    word2idx = {}
    vectors = []
    with open(f'{glove_dir}/{glove_name_d}.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = np.array(vectors)

    if padding:
        # add padding
        word2idx = {k:(v+1) for k,v in word2idx.items()}
        vectors = np.vstack([np.zeros(vectors.shape[1]),vectors])

    #save files
    pickle.dump(words, open(glove_words_path, 'wb'))
    pickle.dump(word2idx, open(glove_idx_path, 'wb'))
    np.save(glove_vectors_path, vectors)

    return glove_idx_path, glove_vectors_path


def load_glove_array(
    dir = get_glove_dir(),
    glove_name_d='glove.6B.50d',
    padding=True,
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
    _, _, glove_idx_path, glove_vectors_path = \
        get_glove_files(dir, glove_name_d, padded=padding)
    
    #load files
    #words = pickle.load(open(glove_words_path, 'rb'))
    word2idx = pickle.load(open(glove_idx_path, 'rb'))
    vectors = np.load(glove_vectors_path)

    # the order should be correct as created in parse_and_save_glove

    return word2idx, vectors


def get_glove(
    glove_dir = get_glove_dir(),
    glove_name = 'glove.6B',
    glove_dim = 50,
    **kwargs
):
    """
    Download, parse and load GloVe word embeddings.
    only downloads if not already downloaded | override kwarg: force_download
    only parses if not already parsed | override kwarg: force_redo
    Parameters
    ----------
    glove_dir : str
        Path to the directory where the GloVe files are stored. default: '...src/data/GloVe'
    glove_name : str
        Name of the GloVe file. default: 'glove.6B'
    glove_dim : int
        Dimensionality of the GloVe word embeddings. default: 50
    """
    # download glove
    download_glove(glove_name, glove_dir, **kwargs)

    # parse glove
    glove_name_d = f'{glove_name}.{glove_dim}d'
    parse_and_save_glove_array(glove_dir, glove_name_d, **kwargs)

    # load glove
    word2idx, vectors = load_glove_array(glove_dir, glove_name_d)

    return word2idx, vectors