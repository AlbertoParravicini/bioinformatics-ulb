import string
import random

def generate_string(size=10, chars=string.ascii_uppercase + string.digits):
    """
    Generates a random string.
    Taken from: http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    
    Parameters
    ----------
    k: int
        Size of the string to be generated
    
    chars: array-like
        Set of characters to use to generate the string
    
    Returns
    ----------
    string
    """
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))