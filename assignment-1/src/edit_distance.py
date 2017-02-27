import numpy as np
import editdistance
import utils
import random

aminoacid_names = "ARNDCEQGHILKMFPSTWYV"
print("Number of aminoacids:", len(aminoacid_names))

gap_penalty = -1

min_string_size = 20
max_string_size = 70


def edit_levenshtein(c1, c2):
    """
    Default edit function cost introduced by Levenshtein
    
    Parameters 
    ----------
    c1, c2: any object on which equality is defined.
    
    Returns
    ----------
    int
        0 if c1 == c2 (i.e. no substitution is needed),
        -1 otherwise (i.e. a substitution is needed)
    """
    return 0 if c1 == c2 else -1
    

def edit_distance(s1, s2, gap_penalty = -1, edit_function = edit_levenshtein):

    """
    Compute the edit distance between 2 strings "s1" and "s2", 
    i.e. the number of character deletions, insertions and substitutions 
    required to turn "s1" into "s2".
    
    Parameters 
    ----------
    s1, s2: array-like
        
    gap_penalty: int, optional
        The penalty factor assigned to character deletions and insertions.
        
    edit_function: function, optional
        The function that is used to compute the cost of a character subtitution.
        
    
    Returns 
    ----------
    int
        The edit distance between s1 and s2
    """
    n_row= len(s1) + 1
    n_col = len(s2) + 1
    edit_matrix = np.zeros((n_row, n_col))
    
    for i in range(n_row):
        edit_matrix[i, 0] = i * gap_penalty
                    
    for j in range(n_col):
        edit_matrix[0, j] = j * gap_penalty
                       
    for i in range(1, n_row):
        for j in range(1, n_col):
            x_gap = edit_matrix[i - 1, j] + gap_penalty
            y_gap = edit_matrix[i, j - 1] + gap_penalty
            mut = edit_matrix[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1])
            edit_matrix[i, j] = max(x_gap, y_gap, mut)
            
    return -edit_matrix[len(s1), len(s2)]
    


# Perform a quick tets to see if the edit distance is ok!        
n_test = 100
for _ in range(n_test):
    s1 = utils.generate_string(random.randint(min_string_size, max_string_size), aminoacid_names)
    s2 = utils.generate_string(random.randint(min_string_size, max_string_size), aminoacid_names)
    assert edit_distance(s1, s2) == editdistance.eval(s1, s2), "Edit distance is wrong!"
                  
            