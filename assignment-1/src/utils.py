import string
import random
from Bio.SubsMat import MatrixInfo


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
	


def sub_matrices_distance(c1, c2, matrix=MatrixInfo.pam120):
    """
    Get the substitution score for c1 and c2 
    according to the provided substitution matrix.

    Parameters
    ----------
    c1, c2: char

    matrix: Bio.SubsMat.MatrixInfo.available_matrix, optional
        The substitution matrix to be used, among the ones available in Bio.SubsMat.MatrixInfo

    Returns 
    ----------
    int  
        the score for substituting c1 with c2.
    """
    return matrix[(c1, c2)] if (c1, c2) in matrix else matrix[(c2, c1)]
	
def gap_function(gap_penalty, gap_opening_penalty, k):
	"""
	Compute the cost of a gap given the input parameters.
	
	Parameters
    ----------
	gap_penalty: int
		Cost for extending a gap.
		
	gap_opening_penalty: int
		Additional cost for opening a gap.
		
	k: int
		Lenght of the current gap
	"""
	return gap_opening_penalty + (k * gap_penalty)
    

class Alignment:
    """
    Class used to store an alignment of 2 sequences, stored as s1 and s2. 
    It also contains a matching string, where for each aminoacid in s1 and s2, we have:\n
        ":" Perfect match of aminoacids.\n
        "." A substitution/mutation occured.\n
        " " A gap was introduced.
    It is possible to add the score of the alignment, if available.
    """

    def __init__(self, s1="", s2="", match="", score=0):
        self.s1 = s1
        self.s2 = s2
        self.match = match
        self.score = 0
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Alignment(s1 = self.s1 + other.s1, s2 = self.s2 + other.s2, match = self.match + other.match)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.s1 += other.s1
            self.s2 += other.s2
            self.match += other.match
        else:
             return NotImplemented

    def __str__(self):
        return self.s1 + "\n" + self.match + "\n" + self.s2 + "\n\n----- SCORE:" + str(self.score)

    def __repr__(self):
        return str(self)
    
	