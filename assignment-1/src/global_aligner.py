import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import random
import pandas as pd

import timeit
import time 

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
    

def global_aligner(s1, s2, gap_penalty = -1, edit_function=sub_matrices_distance, matrix=MatrixInfo.pam120, semiglobal=False):
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
        
    semiglobal: bool
        Set to false to penalize the sequences for not being aligned at the start.
        If true, don't penalize gaps at the beginning of the alignment.

    Returns 
    ----------
    int
        The edit distance between s1 and s2

    float64 np.matrix
        The alignment matrix of s1 and s2
    """
    n_row= len(s1) + 1
    n_col = len(s2) + 1
    edit_matrix = np.zeros((n_row, n_col))
    
    for i in range(n_row):
        edit_matrix[i, 0] = i * (0 if semiglobal else gap_penalty)
                    
    for j in range(n_col):
        edit_matrix[0, j] = j * (0 if semiglobal else gap_penalty)
                       
    for i in range(1, n_row):
        for j in range(1, n_col):
            s1_gap = edit_matrix[i - 1, j] + gap_penalty
            s2_gap = edit_matrix[i, j - 1] + gap_penalty
            mut = edit_matrix[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            edit_matrix[i, j] = max(s1_gap, s2_gap, mut)
            
    # If semiglobal alignment, get the best value on the last row.
    align_score = max(edit_matrix[len(s1), :]) if semiglobal else edit_matrix[len(s1), len(s2)]
    return [align_score, edit_matrix]
    

def backtrack_matrix(s1, s2, input_matrix, gap_penalty=-1, edit_function=sub_matrices_distance, matrix=MatrixInfo.pam120, semiglobal=False):
    i = len(s1)
    j = len(s2) 
    aligned_s1 = ""
    aligned_s2 = ""    
    match_sequence = ""
      
    while i > 0 or j > 0:
        val = input_matrix[i, j]
        s1_gap = input_matrix[i - 1, j] + (0 if (semiglobal == True and j == 0) else gap_penalty)
        s2_gap = input_matrix[i, j - 1] + (0 if (semiglobal == True and i == 0) else gap_penalty)
        mut = input_matrix[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)

        # Match s1 to a gap, move vertically
        if i > 0 and val == s1_gap:
            aligned_s1 = s1[i - 1] + aligned_s1
            aligned_s2 = "-" + aligned_s2
            match_sequence = " " + match_sequence
            i -= 1
        # Match s2 to a gap, move horizontally
        elif j > 0 and val == s2_gap:
            aligned_s1 = "-" + aligned_s1
            aligned_s2 = s2[j - 1] + aligned_s2
            match_sequence = " " + match_sequence
            j -= 1
        # Substitution, diagonal movement
        elif i > 0 and j > 0 and val == mut:
            aligned_s1 = s1[i - 1] + aligned_s1
            aligned_s2 = s2[j - 1] + aligned_s2
            match_sequence = (":" if s1[i - 1] == s2[j - 1] else ".") + match_sequence
            i -= 1
            j -= 1
        else:
            raise ValueError("val={0}, but we have s1_gap={1}, s2_gap={2}, mut={3}".format(val, s1_gap, s2_gap, mut))

    return [aligned_s1, aligned_s2, match_sequence]
                
                  
# Load the sequences and test their edit distance
start_time = timeit.default_timer()

for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
   for j, seq_record_j in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
       if i > j :
#            print("Comparing:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
#            print("\t", seq_record_j.id, "-- length:", len(seq_record_j))
            [score, edit_matrix] = global_aligner(seq_record_i.seq,  seq_record_j.seq, gap_penalty=-1, matrix=MatrixInfo.blosum62)
            #[s1_al, s2_al, match_sequence] = backtrack_matrix(seq_record_i.seq, seq_record_j.seq, edit_matrix, gap_penalty=-1, matrix=MatrixInfo.blosum62)
            
#            print(s1_al)
#            print(match_sequence)
#            print(s2_al)
#            print("MY ALIGNER:", score)
#            print("BIOPYTHON ALIGNER", pairwise2.align.globaldx(seq_record_i.seq, seq_record_j.seq, MatrixInfo.blosum62, score_only = True))
#            print("\n")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")




s1 = "THISLINE"
s2 = "ISALIGNED"
[score, edit_matrix] = global_aligner(s1, s2, gap_penalty=-4, matrix=MatrixInfo.blosum62, semiglobal=True)

print(edit_matrix)
print(score)

edit_frame = pd.DataFrame(edit_matrix)
edit_frame.index = list(" " + s1)
edit_frame.columns = list(" " + s2)

[s1_al, s2_al, mat] = backtrack_matrix(s1, s2, edit_matrix, gap_penalty=-4, matrix=MatrixInfo.blosum62, semiglobal=True)
print(s1_al)
print(mat)
print(s2_al)
