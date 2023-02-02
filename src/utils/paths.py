import os

def get_data_dir():
    """Get the path to the data directory. Create it if it doesn't exist."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_mind_dir():
    """Get the path to the mind directory. Create it if it doesn't exist."""
    mind_dir = os.path.join(get_data_dir(), 'MIND')
    os.makedirs(mind_dir, exist_ok=True)
    return mind_dir

def get_model_dir():
    """Get the path to the model directory. Create it if it doesn't exist."""
    data_dir = get_data_dir()
    model_dir = os.path.join(data_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir