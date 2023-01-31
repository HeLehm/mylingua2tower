import os

def get_data_dir():
    """Get the path to the data directory. Create it if it doesn't exist."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_glove_dir():
    """Get the path to the glove directory. Create it if it doesn't exist."""
    glove_dir = os.path.join(get_data_dir(), 'GloVe')
    os.makedirs(glove_dir, exist_ok=True)
    return glove_dir

def get_bcolz_glove_files(dir,glove_name):
    glove_path = f'{dir}/{glove_name}.txt'
    glove_words_path = f'{dir}/{glove_name}_words.pkl'
    glove_idx_path = f'{dir}/{glove_name}_idx.pkl'
    glove_vectors_path = f'{dir}/{glove_name}.dat'
    return glove_path, glove_words_path, glove_idx_path, glove_vectors_path

def get_np_glove_files(dir,glove_name):
    glove_path = f'{dir}/{glove_name}.txt'
    glove_words_path = f'{dir}/{glove_name}_words.pkl'
    glove_idx_path = f'{dir}/{glove_name}_idx.pkl'
    glove_vectors_path = f'{dir}/{glove_name}.npy'
    return glove_path, glove_words_path, glove_idx_path, glove_vectors_path

def get_mind_dir():
    """Get the path to the mind directory. Create it if it doesn't exist."""
    mind_dir = os.path.join(get_data_dir(), 'MIND')
    os.makedirs(mind_dir, exist_ok=True)
    return mind_dir