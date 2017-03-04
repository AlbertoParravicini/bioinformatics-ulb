import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import pandas as pd
from itertools import compress

import global_aligner_2
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




##########################################
# LOCAL ALIGNER SCORE ####################
##########################################

def local_aligner_score(s1, s2, gap_penalty=-1, gap_opening_penalty=-10, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120):
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
            s1_gap = max([S[i - k, j] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
            s2_gap = max([S[i, j - k] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
            mut = S[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            # In the local aligner, don't accept negative scores!
            S[i, j] = max(s1_gap, s2_gap, mut, 0)

            if S[i, j] >= score_max:
                score_max = S[i, j]
                i_max = i
                j_max = j
            # Write in the matrix the movement that lead to that cell, as a string.
            # e.g. "HV" means that horizontal and vertical movements were the
            # best.
            # In local alignment, "X" means that 0 was the maximum value, and all the movements gave a negative score.
            # The backtracking will stop when an "X" is encountered.
            backtrack_matrix.set_value(i, j, "".join(check_argmax([s1_gap, s2_gap, mut, 0])))
    
    return [score_max, S, backtrack_matrix, i_max, j_max]




##########################################
# BACKTRACK ##############################
##########################################

def reconstruct_sequence(s1, s2, S, backtrack_matrix, gap_penalty, gap_opening_penalty, edit_function, matrix):

    coordinate_list = []
    
    [i, j] = backtrack_matrix.shape
    i-=1
    j-=1
          
    while i > 0 or j > 0:
        val = S[i, j]
        # Consider 0 to handle the first row/column
        s1_gap = 0 if i == 0 else max([S[i - k, j] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
        s2_gap = 0 if j == 0 else max([S[i, j - k] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
        mut = S[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)

        # Append the current location to the coordinate list.
        coordinate_list.append([i, j])
        # If a 0 is found, interrupt the traceback
        if val == 0:
            break
        # Match s1 to a gap, move vertically
        elif i > 0 and val == s1_gap:
            i -= 1
        # Match s2 to a gap, move horizontally
        elif j > 0 and val == s2_gap:
            j -= 1
        # Substitution, diagonal movement
        elif i > 0 and j > 0 and val == mut:
            i -= 1
            j -= 1
        else:
            raise ValueError("val={0}, but we have s1_gap={1}, s2_gap={2}, mut={3}".format(val, s1_gap, s2_gap, mut))
    
    coordinate_list.reverse()    
    return coordinate_list

def update_score_matrix(s1, s2, S, coordinate_list, backtrack_matrix, gap_penalty, gap_opening_penalty, edit_function, matrix):
    
    for [i, j] in coordinate_list:    
        # Set the current value to 0 in the optimal matching sequence.
        S[i, j] = 0
        backtrack_matrix.iloc[i, j] = "X"
        # Recompute the values below S[i, j]
        if j > 0 and i > 0:
            # Update the values below the current position.
            for i_i in range(i+1, len(s1)+1):
                # Store the old value of the cell, as if it doesn't change we can stop to update S.
                old_val = S[i_i, j]
                
                s1_gap = max([S[i_i - k, j] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i_i+1)])
                s2_gap = max([S[i_i, j - k] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
                mut = S[i_i - 1, j - 1] + edit_function(s1[i_i - 1], s2[j - 1], matrix=matrix)
                # In the local aligner, don't accept negative scores!
                if backtrack_matrix.iloc[i_i, j] != "X":
                    S[i_i, j] = max(s1_gap, s2_gap, mut, 0)
                if S[i_i, j] == old_val:
                    break
            # Update the values on the right the current position.
            for j_j in range(j+1, len(s2)+1):
                # Store the old value of the cell, as if it doesn't change we can stop to update S.
                old_val = S[i, j_j]
                
                s1_gap = max([S[i - k, j_j] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
                s2_gap = max([S[i, j_j - k] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j_j+1)])
                mut = S[i - 1, j_j - 1] + edit_function(s1[i - 1], s2[j_j - 1], matrix=matrix)
                # In the local aligner, don't accept negative scores!
                if backtrack_matrix.iloc[i, j_j] != "X":
                    S[i, j_j] = max(s1_gap, s2_gap, mut, 0)
                if S[i, j_j] == old_val:
                    break
    for i in range(coordinate_list[-1][0]+1, len(s1)+1):
        for j in range(coordinate_list[-1][1]+1, len(s2)+1):
            s1_gap = max([S[i - k, j] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
            s2_gap = max([S[i, j - k] + utils.gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
            mut = S[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            if backtrack_matrix.iloc[i, j] != "X":
                S[i, j] = max(s1_gap, s2_gap, mut, 0)

##########################################
# LOCAL ALIGNER ##########################
##########################################

def local_aligner(s1, s2, gap_penalty=-1, gap_opening_penalty=-10, k=1, sub_alignments_num=1, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120):
    alignments = []
    
    # Build the initial score matrix.
    [score, S, backtrack_matrix, i_max, j_max] = local_aligner_score(s1, s2, gap_penalty=gap_penalty, gap_opening_penalty=gap_opening_penalty, edit_function=edit_function, matrix=matrix)
    for n in range(sub_alignments_num):
        align_list_n = global_aligner_2.backtrack_sequence_rec(s1[:i_max], s2[:j_max], backtrack_matrix.iloc[:i_max+1, :j_max+1], k=k)
        
        # Add the alignment scores to each alignment
        for align_i in align_list_n:
            align_i.score = score
        # Add the alignments to the overall list of alignments
        alignments += align_list_n
        
        # Update the score matrix to get more subalignments.
        # Small optimization: done only if sub_alignments_num > 1
        if sub_alignments_num > 1:
            # Update the score matrix to get more subalignments.
            # Get the coordinates of one best matching
            coordinate_list = reconstruct_sequence(s1, s2, S, backtrack_matrix.iloc[:i_max+1, :j_max+1], gap_penalty, gap_opening_penalty, edit_function, matrix)
            update_score_matrix(s1, s2, S, coordinate_list, backtrack_matrix, gap_penalty, gap_opening_penalty, edit_function, matrix)

            # Find the new maximum value in the matrix.
            [i_max, j_max] = np.unravel_index(np.argmax(S), S.shape)
            score = S[i_max, j_max]
            if i_max == 0 and j_max == 0:
                break
       
    return alignments
    

    
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
#            [score, edit_matrix, backtrack_matrix, i_max, j_max] = local_aligner_score(seq_record_i.seq,  seq_record_j.seq, gap_penalty=-1, matrix=MatrixInfo.blosum62)
            align_list = local_aligner(seq_record_i.seq,  seq_record_j.seq, -4, 0, 4, 7, matrix=MatrixInfo.blosum62)
#            align_list = backtrack_sequence_rec(seq_record_i.seq, seq_record_j.seq, backtrack_matrix, k=1)

#            for p in align_list:
#               print(str(p) + "\n\n")
            print("DONE")
#            print("MY ALIGNER:", score)
            print("\n")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")


s1 = "THISLINE"
s2 = "ISALIGNED"
#[score, edit_matrix, backtrack_matrix, i_max, j_max] = local_aligner_score(s1, s2, gap_penalty=-4, gap_opening_penalty=0, matrix=MatrixInfo.blosum62)
#print(edit_matrix)
#print(backtrack_matrix)
#
#edit_frame = pd.DataFrame(edit_matrix)
#edit_frame.index = list(" " + s1)
#edit_frame.columns = list(" " + s2)
#
#print("\nSCORE:", score)
#
#align_list = global_aligner_2.backtrack_sequence_rec(s1[:i_max], s2[:j_max], backtrack_matrix.iloc[:i_max+1, :j_max+1], k=3)
#
#for p in align_list:
#    print(str(p) + "\n")
#print("DONE")



align_list = local_aligner(s1, s2, -4, 0, 4, 7, matrix=MatrixInfo.blosum62)
for align_i in align_list:
    print(align_i)



