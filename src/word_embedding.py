# script for GloVe word embedding
# source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
import os
import io
import zipfile
import requests

from glob import glob
import bcolz
import pickle
import numpy as np
import torch.nn as nn
import torch

from .paths import get_glove_dir, get_glove_files

# TODO: index 0 = 0


def dowload_glove(
    glove_name = 'glove.6B',
    path=get_glove_dir(),
    force_download=False
):
    """Download GloVe word embeddings."""
    download_url = f'https://nlp.stanford.edu/data/{glove_name}.zip'

    #check if file already exists
    path_wildcard = path + f'/{glove_name}.*d.txt'
    if not force_download and len(glob(path_wildcard)) > 0:
        print('GloVe files already exist. Delete them or use force_download=True to download again.')
        return
    print('Downloading GloVe word embeddings...')


    r = requests.get(download_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    z.close()

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
        get_glove_files(dir,glove_name_d)
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
        get_glove_files(dir,glove_name_d)
    
    #load files
    words = pickle.load(open(glove_words_path, 'rb'))
    word2idx = pickle.load(open(glove_idx_path, 'rb'))
    vectors = bcolz.open(glove_vectors_path)[:]

    # the order should be correct as created in parse_and_save_glove

    return word2idx, vectors

def get_glove_embedding_matrix(
    word2idx,
    idx2emb,
    target_vocab,
    emb_dim=50,
):
    """
    Get the embedding matrix for a given vocabulary.
    Parameters
    ----------
    glove : dict
        GloVe word embeddings.
    target_vocab : list
        Vocabulary for which the embedding matrix is created.
    emb_dim : int
        Dimension of the GloVe word embeddings.
    Returns
    -------
    weights_matrix : numpy.ndarray
        Embedding matrix of shape (len(target_vocab), emb_dim).
    """
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i] = idx2emb[word2idx[word]]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    return weights_matrix

def create_nn_embedding_layer_from_matrix(weights_matrix : np.ndarray, trainable=False):
    """
    Create embedding layer for neural network.
    """
    #extend embedding matrix by one row for padding
    weights_matrix = np.vstack((np.zeros(weights_matrix.shape[1]), weights_matrix))
    weights_matrix = torch.from_numpy(weights_matrix).float()

    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def create_embedding_layer(
    glove_name = 'glove.6B',
    dim = 50,
    path=get_glove_dir(),
    force_download=False,
    force_redo=False,
    target_vocab=None,
):
    """
    Create embedding layer for neural network.
    Parameters
    ----------
    glove_name : str
        Name of the GloVe file.
    dim : int
        Dimension of the GloVe word embeddings.
    path : str
        Path to the directory where the GloVe files are stored.
    force_download : bool
        If True, overwrite existing files.
        If False, use existing files if they exist.
    force_redo : bool
        If True, overwrite existing files.
        If False, use existing files if they exist.
    target_vocab : list
        Vocabulary for which the embedding matrix is created.
    Returns
    -------
    emb_layer : torch.nn.Embedding
    word2idx : dict
        Dictionary moving from words to indices.
        NOTE: the index in then embedding matrix is index + 1 (padding = 0)
    """
    # download glove file
    dowload_glove(glove_name, path, force_download)

    # parse glove file
    glove_name_d = f'{glove_name}.{dim}d'
    parse_and_save_glove(path, glove_name_d, force_redo)

    # load glove dictionary
    word2idx, idx2emb = load_glove(path, glove_name_d)

    # create embedding matrix
    if target_vocab is None:
        target_vocab = word2idx.keys()
    weights_matrix = get_glove_embedding_matrix(word2idx, idx2emb, target_vocab, dim)

    # create embedding layer
    emb_layer, num_embeddings, embedding_dim = create_nn_embedding_layer_from_matrix(weights_matrix)

    return emb_layer, word2idx

    
    