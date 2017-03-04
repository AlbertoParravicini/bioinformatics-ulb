import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import pandas as pd
from itertools import compress

import timeit


def check_argmax(array):
    """
    Checks which movements are the best, given an array containing 3 scores. 

    Parameters
    ----------
    array: array-like
        numerical array of size 3, it contains the scores relative to character insertion/deletion/subtitution

    Returns
    ----------
    string
        A string of 1 to 3 chars from "V, H, D": a character appears in the return string
        if it is an optimal movement, with maximum score.
    """
    # Check which movements are the best, return it as a list where 1 = max of the list.
    res = [1 if i == max(array) else 0 for i in array]
    return list(compress(["V", "H", "D", "X"], res))


def local_aligner(s1, s2, gap_penalty=-1, gap_opening_penalty=-10, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120):
    
    
    def gap_function(gap_penalty, gap_opening_penalty, k):
        return gap_opening_penalty + (k * gap_penalty)

    n_row = len(s1) + 1
    n_col = len(s2) + 1
    # Creates a matrix where the partial scores are stored.
    S = np.zeros((n_row, n_col))
    # Creates a matrix (stored as DataFrame) where the optimal movements are
    # stored.
    backtrack_matrix = pd.DataFrame("", index=np.arange(n_row), columns=np.arange(n_col))

    # Initialize the first column and row of the matrices.
    # In the local aligner, we stop when a 0 is encountered, which corresponds to an "X"
    for i in range(n_row):
        backtrack_matrix.set_value(i, 0, "X")

    for j in range(n_col):
        backtrack_matrix.set_value(0, j, "X")
    
    # small optimization: keep track of the maximum score encountered so far, and its indices.
    score_max = 0
    i_max = 0
    j_max = 0
    
    for i in range(1, n_row):
        for j in range(1, n_col):
            # Compute the possible movements, and then keeps the best.
            s1_gap = max([S[i - k, j] + gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
            s2_gap = max([S[i, j - k] + gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
            mut = S[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            # In the local aligner, don't accept negative scores!
            S[i, j] = max(s1_gap, s2_gap, mut, 0)

            if S[i, j] > score_max:
                score_max = S[i, j]
                i_max = i
                j_max = j
            # Write in the matrix the movement that lead to that cell, as a string.
            # e.g. "HV" means that horizontal and vertical movements were the
            # best.
            # In local alignment, "X" means that 0 was the maximum value, and all the movements gave a negative score.
            # The backtracking will stop when an "X" is encountered.
            backtrack_matrix.set_value(i, j, "".join(check_argmax([s1_gap, s2_gap, mut, 0])))
    
    return [score_max, S, backtrack_matrix.iloc[:i_max+1, :j_max+1]]


##########################################
# TEST ###################################
##########################################


start_time = timeit.default_timer()
# Load the sequences and test their edit distance
for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
    for j, seq_record_j in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
        if i > j:
            print("Comparing:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
            print("\t", seq_record_j.id, "-- length:", len(seq_record_j))
            [score, edit_matrix, backtrack_matrix] = local_aligner(seq_record_i.seq,  seq_record_j.seq, gap_penalty=-1, matrix=MatrixInfo.blosum62)
#            align_list = backtrack_sequence_rec(seq_record_i.seq, seq_record_j.seq, backtrack_matrix, k=1)

#            for p in align_list:
#               print(str(p) + "\n\n")
            print("DONE")
            print("MY ALIGNER:", score)
            print("\n")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")


s1 = "THISLINE"
s2 = "ISALIGNED"
[score, edit_matrix, backtrack_matrix] = local_aligner(s1, s2, gap_penalty=-8, gap_opening_penalty=0, matrix=MatrixInfo.blosum62)
print(edit_matrix)
print(backtrack_matrix)

edit_frame = pd.DataFrame(edit_matrix)
edit_frame.index = list(" " + s1)
edit_frame.columns = list(" " + s2)

print("\nSCORE:", score)

#align_list = semiglobal_backtrack(s1, s2, edit_matrix, backtrack_matrix)

#for p in align_list:
#    print(str(p) + "\n")
#print("DONE")


