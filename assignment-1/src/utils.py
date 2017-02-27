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
	

class Alignment:

    def __init__(self, s1="", s2="", match=""):
        self.s1 = s1
        self.s2 = s2
        self.match = match
    
    def __add__(self, other):
        return Alignment(s1 = self.s1 + other.s1, s2 = self.s2 + other.s2, match = self.match + other.match)

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.s1 += other.s1
            self.s2 += other.s2
            self.match += other.match
        else:
             return NotImplemented

    def __str__(self):
        if isinstance(other, self.__class__):
            return str(self.s1 + "\n" + self.match + "\n" + self.s2)
        return NotImplemented

    
	