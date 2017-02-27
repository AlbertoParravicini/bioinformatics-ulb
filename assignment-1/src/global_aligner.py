import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import random

aminoacid_names = "ARNDCEQGHILKMFPSTWYV"
print("Number of aminoacids:", len(aminoacid_names))

gap_penalty = -1

min_string_size = 20
max_string_size = 70

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
    

def global_aligner(s1, s2, gap_penalty = -1, edit_function=sub_matrices_distance, matrix=MatrixInfo.pam120):
    """
    Compute the global alignment between 2 aminoacid sequences "s1" and "s2".
    
    Parameters 
    ----------
    s1, s2: string
        The two input aminoacid sequences on which the edit distance is computed.
        
    gap_penalty: int, optional
        The penalty factor assigned to matching an aminoacid to a gap character.
        It should be a NEGATIVE integer.
        
    edit_function: function, optional
        The function that is used to compute the cost of an aminoacid subtitution.
        
    matrix: Bio.SubsMat.MatrixInfo.available_matrix, optional
        The substitution matrix to be used, among the ones available in Bio.SubsMat.MatrixInfo.
        It is used by edit_function, if needed.
        
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
            mut = edit_matrix[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            edit_matrix[i, j] = max(x_gap, y_gap, mut)
            
    return [edit_matrix[len(s1), len(s2)], edit_matrix]
    


                  
# Load the sequences and test their edit distance
#for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
#    for j, seq_record_j in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
#        if i > j :
#            print("Comparing:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
#            print("\t", seq_record_j.id, "-- length:", len(seq_record_j))
#            print("MY ALIGNER:", global_aligner(seq_record_i.seq, seq_record_j.seq))
#            print("BIOPYTHON ALIGNER", pairwise2.align.globaldx(seq_record_i.seq, seq_record_j.seq, MatrixInfo.pam120, score_only = True))
#            print("\n")


s1 = "THISLINE"
s2 = "ISALIGNED"
[score, edit_matrix] = global_aligner(s1, s2, gap_penalty=-4, matrix=MatrixInfo.blosum62)
print(edit_matrix)
print(score)

