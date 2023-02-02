# script for GloVe word embedding
# source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
import io
import zipfile
import requests
import torch
import torch.nn as nn
import numpy as np

from  typing import Callable

from glob import glob

from ..paths import get_glove_dir

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

def create_nn_embedding_layer_from_matrix(weights_matrix : np.ndarray, trainable=False, padding=True):
    """
    Create embedding layer for neural network.
    """
    if padding:
        #extend embedding matrix by one row for padding
        weights_matrix = np.vstack((np.zeros(weights_matrix.shape[1]), weights_matrix))
        weights_matrix = torch.from_numpy(weights_matrix).float()

    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0 if padding else None)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def _create_torch_embedding_layer(
    glove_name = 'glove.6B',
    dim = 50,
    glove_path=get_glove_dir(),
    force_download=False,
    force_redo=False,
    target_vocab=None,
    parse_and_save_glove: Callable = None,
    load_glove: Callable = None,
    padding = True,
    trainable = False,
    **kwargs
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
    dowload_glove(glove_name, glove_path, force_download)

    # parse glove file
    glove_name_d = f'{glove_name}.{dim}d'
    parse_and_save_glove(glove_path, glove_name_d, force_redo)

    # load glove dictionary
    word2idx, idx2emb = load_glove(glove_path, glove_name_d)

    # create embedding matrix
    if target_vocab is None:
        target_vocab = word2idx.keys()
    weights_matrix = get_glove_embedding_matrix(word2idx, idx2emb, target_vocab, dim)

    # create embedding layer
    emb_layer, num_embeddings, embedding_dim = create_nn_embedding_layer_from_matrix(weights_matrix, trainable=trainable, padding=padding)

    # offset word2idx by 1 (because of padding)
    word2idx = {word: idx + 1 for word, idx in word2idx.items()}

    return emb_layer, word2idx