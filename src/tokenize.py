from nltk.tokenize import wordpunct_tokenize

def tokenize(text):
    """
    Tokenizes the lower cased text.
    Parameters
    ----------
    text : str
        Text to tokenize.
    """
    return wordpunct_tokenize(text.lower())