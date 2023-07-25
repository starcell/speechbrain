import re

def string_normalize(string):
    """
    This function normalizes a given string according to
    the normalization rule
    The normalization rule removes "/" indicating filler words,
    removes "+" indicating repeated words,
    removes all punctuation marks,
    removes non-speech symbols,
    and extracts orthographic transcriptions.

    Arguments
    ---------
    string : str
        The string to be normalized

    Returns
    -------
    str
        The string normalized according to the rules

    """
    # extracts orthographic transcription
    string = re.sub(r"\(([^)]*)\)\/\(([^)]*)\)", r"\1", string)
    # removes non-speech symbols
    string = re.sub(r"n/|b/|o/|l/|u/", "", string)
    # removes punctuation marks
    string = re.sub(r"[+*/.?!~(),]", "", string)
    # removes extra spaces
    string = re.sub(r"\s+", " ", string)
    string = string.strip()

    return string