import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import pandas as pd
from itertools import compress

import timeit

def global_aligner_2(s1, s2, gap_penalty=-1, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120, semiglobal=False):
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
        The alignment score of s1 and s2

    float64 np.matrix
        The alignment matrix of s1 and s2

    pandas.DataFrame
        The backtrack matrix, which shows the optimal matching "movements" for each cell.
    """
    n_row = len(s1) + 1
    n_col = len(s2) + 1
    # Creates a matrix where the partial scores are stored.
    edit_matrix = np.zeros((n_row, n_col))
    # Creates a matrix (stored as DataFrame) where the optimal movements are
    # stored.
    backtrack_matrix = pd.DataFrame("", index=np.arange(n_row), columns=np.arange(n_col))

    # Initialize the first column and row of the matrices.
    for i in range(n_row):
        edit_matrix[i, 0] = i * (0 if semiglobal else gap_penalty)
        backtrack_matrix.set_value(i, 0, "V")

    for j in range(n_col):
        edit_matrix[0, j] = j * (0 if semiglobal else gap_penalty)
        backtrack_matrix.set_value(0, j, "H")

    # Set the first cell of the backtrack matrix to "X", as an end-marker.
    backtrack_matrix.set_value(0, 0, "X")

    for i in range(1, n_row):
        for j in range(1, n_col):
            # Compute the possible movements, and then keeps the best.
            s1_gap = edit_matrix[i - 1, j] + gap_penalty
            s2_gap = edit_matrix[i, j - 1] + gap_penalty
            mut = edit_matrix[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            edit_matrix[i, j] = max(s1_gap, s2_gap, mut)

            # Write in the matrix the movement that lead to that cell, as a string.
            # e.g. "HV" means that horizontal and vertical movements were the
            # best.
            backtrack_matrix.set_value(i, j, "".join(check_argmax([s1_gap, s2_gap, mut])))
    
    # If semiglobal alignment, get the best value on the last row.
    align_score = max(edit_matrix[len(s1), :]) if semiglobal else edit_matrix[len(s1), len(s2)]
    return [align_score, edit_matrix, backtrack_matrix]


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
    return list(compress(["V", "H", "D"], res))


def backtrack_sequence_rec(s1, s2, backtrack_matrix, k=1):
    """
    Given 2 strings s1 and s2, and a backtrack matrix produced by the global aligner, finds k optimal alignments. 

    Parameters
    ----------
    s1, s2: string
        The 2 strings to be aligned. 
    
    backtrack_matrix: pandas.DataFrame
        Matrix containing the best movements, produced by the global aligner applied to s1 and s2. 
    
    k: int, optional
        The number of best alignment to return, in case there are more alignments with the same optimal score.

    Returns
    ----------    
    array of Alignment
        A list of k alignments, of class utils.Alignment
    """
    i = len(s1)
    j = len(s2)

    align_list = []

    try:
        # For all the possible moves in a cell, in case there's more than 1.
        for mov in backtrack_matrix.iloc[i, j]:
            # Only if we don't already have k sequences, so we don't waste time.
            if len(align_list) < k:
                # For each move, we apply recursively the backtraking to a subset of the matrix,
                # and at the end append the current matching, if we have less than k alignments.
                if mov == "V":
                    for align_i in backtrack_sequence_rec(s1[:i - 1], s2[:j], backtrack_matrix.iloc[:i, :j + 1], k=k):
                        if len(align_list) < k:
                            align_list.append(
                                align_i + utils.Alignment(s1[i - 1], "-", " "))
                        else: break

                elif mov == "H":
                    for align_i in backtrack_sequence_rec(s1[:i], s2[:j - 1], backtrack_matrix.iloc[:i + 1, :j], k=k):
                        if len(align_list) < k:
                            align_list.append(
                                align_i + utils.Alignment("-", s2[j - 1], " "))
                        else: break

                elif mov == "D":
                    for align_i in backtrack_sequence_rec(s1[:i - 1], s2[:j - 1], backtrack_matrix.iloc[:i, :j], k=k):
                        if len(align_list) < k:
                            align_list.append(
                                align_i + utils.Alignment(s1[i - 1], s2[j - 1],  (":" if s1[i - 1] == s2[j - 1] else ".")))
                        else: break
                # End of the recursion, stop if we reach the first cell.
                elif mov == "X":
                    return [utils.Alignment("", "", "")]

                else: raise ValueError("val={0} isn't a valid movement!".format(mov))
    except IndexError:
        print("i:", i, " -- j:", j)
        raise

    return align_list


def semiglobal_backtrack(s1, s2, score_matrix, backtrack_matrix, k=1):
    """
    Backtrack function used for semiglobal alignments.
    Given 2 strings s1 and s2, and a backtrack matrix produced by the global aligner, finds k optimal alignments. 

    Parameters
    ----------
    s1, s2: string
        The 2 strings to be aligned. 
    
    backtrack_matrix: pandas.DataFrame
        Matrix containing the best movements, produced by the global aligner applied to s1 and s2. 
    
    k: int, optional
        The number of best alignment to return, in case there are more alignments with the same optimal score.

    Returns
    ----------    
    array of Alignment
        A list of k alignments, of class utils.Alignment
    """
    # Find where the maximum score is, in the last row of the matrix.
    j_max = np.argmax(score_matrix[len(s1), :])
    # Apply backtracking only to the relevant part of the matrix.
    align_list = backtrack_sequence_rec(s1, s2[:j_max], backtrack_matrix.iloc[:, :j_max+1], k)
    # Append the remaining part of the sequence to the end
    align_list = [align_i + utils.Alignment("-" * (len(s2) - j_max), s2[j_max:], " "* (len(s2) - j_max)) for align_i in align_list]
    return align_list


#def global_aligner_affine_penalty_2(s1, s2, gap_penalty=-1, gap_opening_penalty=-10, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120, semiglobal=False):
#    """
#    Compute the global alignment between 2 aminoacid sequences "s1" and "s2".
#
#    Parameters 
#    ----------
#    s1, s2: string
#        The two input aminoacid sequences on which the edit distance is computed.
#
#    gap_penalty: int, optional
#        The penalty factor assigned for matching an aminoacid to a gap character.
#        It should be a NEGATIVE integer.
#        
#    gap_opening_penalty: int, optional
#        The penalty factor assigned for starting a gap sequence.
#        It should be a NEGATIVE integer.    
#
#    edit_function: function, optional
#        The function that is used to compute the cost of an aminoacid subtitution.
#
#    matrix: Bio.SubsMat.MatrixInfo.available_matrix, optional
#        The substitution matrix to be used, among the ones available in Bio.SubsMat.MatrixInfo.
#        It is used by edit_function, if needed.
#
#    semiglobal: bool
#        Set to false to penalize the sequences for not being aligned at the start.
#        If true, don't penalize gaps at the beginning of the alignment.
#
#    Returns 
#    ----------
#    int
#        The alignment score of s1 and s2
#
#    float64 np.matrix
#        The alignment matrix of s1 and s2
#
#    pandas.DataFrame
#        The backtrack matrix, which shows the optimal matching "movements" for each cell.
#    """
#    
#    n_row = len(s1) + 1
#    n_col = len(s2) + 1
#    # Creates a matrix where the partial scores are stored.
#    S = np.zeros((n_row, n_col))
#    # Additional matrices used to handle the affine gap penalty.
#    V = np.zeros((n_row, n_col))
#    W = np.zeros((n_row, n_col))
#    # Creates a matrix (stored as DataFrame) where the optimal movements are
#    # stored.
#    backtrack_matrix = pd.DataFrame("", index=np.arange(n_row), columns=np.arange(n_col))
#
#    gap_opening_penalty += gap_penalty
#    # Initialize the first column and row of the matrices.
#    S[0, 0] = 0
#    V[0, 0] = 0
#    W[0, 0] = 0
#    for i in range(1, n_row):
#        V[i, 0] = 0 if semiglobal else max(S[i-1, 0] + gap_opening_penalty, V[i-1, 0] + gap_penalty)
#        W[i, 0] = -np.Infinity
#        S[i, 0] = i * (0 if semiglobal else gap_penalty)
#        backtrack_matrix.set_value(i, 0, "V")
#
#    for j in range(1, n_col):
#        V[0, j] = -np.Infinity
#        W[0, j] = 0 if semiglobal else max(S[0, j-1] + gap_opening_penalty, W[0, j-1] + gap_penalty)
#        S[0, j] = j * (0 if semiglobal else gap_penalty)
#        backtrack_matrix.set_value(0, j, "H")
#    
#    # Set the first cell of the backtrack matrix to "X", as an end-marker.
#    backtrack_matrix.set_value(0, 0, "X")
#    
#
#    for i in range(1, n_row):
#        for j in range(1, n_col):
#            # Compute the possible movements, and then keeps the best.
#            V[i, j] = max(S[i-1, j] + gap_opening_penalty, V[i-1, j] + gap_penalty)
#            W[i, j] = max(S[i, j-1] + gap_opening_penalty, W[i, j-1] + gap_penalty)
#            mut_cost = edit_function(s1[i-1], s2[j-1], matrix=matrix)
#            S[i, j] = max(V[i, j], W[i, j], S[i-1, j-1] + mut_cost)
#
#            # Write in the matrix the movement that lead to that cell, as a string.
#            # e.g. "HV" means that horizontal and vertical movements were the
#            # best.
#            backtrack_matrix.set_value(i, j, "".join(check_argmax([V[i, j], W[i, j], mut_cost])))
#    
#    # If semiglobal alignment, get the best value on the last row.
#    align_score = max(S[len(s1), :]) if semiglobal else S[len(s1), len(s2)]
#    return [align_score, S, backtrack_matrix]


def global_aligner_affine_penalty(s1, s2, gap_penalty=-1, gap_opening_penalty=-10, edit_function=utils.sub_matrices_distance, matrix=MatrixInfo.pam120, semiglobal=False):
        """
    Compute the global alignment between 2 aminoacid sequences "s1" and "s2".

    Parameters 
    ----------
    s1, s2: string
        The two input aminoacid sequences on which the edit distance is computed.

    gap_penalty: int, optional
        The penalty factor assigned to matching an aminoacid to a gap character.
        It should be a NEGATIVE integer.
        
    gap_opening_penalty: int, optional
        The penalty factor assigned for starting a gap sequence.
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
        The alignment score of s1 and s2

    float64 np.matrix
        The alignment matrix of s1 and s2

    pandas.DataFrame
        The backtrack matrix, which shows the optimal matching "movements" for each cell.
    """
    
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
    for i in range(n_row):
        S[i, 0] = 0 if semiglobal else gap_function(gap_penalty, gap_opening_penalty, i)
        backtrack_matrix.set_value(i, 0, "V")

    for j in range(n_col):
        S[0, j] = 0 if semiglobal else gap_function(gap_penalty, gap_opening_penalty, j)
        backtrack_matrix.set_value(0, j, "H")

    # Manually initialize the first cell
    S[0, 0] = 0
    # Set the first cell of the backtrack matrix to "X", as an end-marker.
    backtrack_matrix.set_value(0, 0, "X")

    for i in range(1, n_row):
        for j in range(1, n_col):
            # Compute the possible movements, and then keeps the best.
            s1_gap = max([S[i - k, j] + gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, i+1)])
            s2_gap = max([S[i, j - k] + gap_function(gap_penalty, gap_opening_penalty, k) for k in range(1, j+1)])
            mut = S[i - 1, j - 1] + edit_function(s1[i - 1], s2[j - 1], matrix=matrix)
            S[i, j] = max(s1_gap, s2_gap, mut)

            # Write in the matrix the movement that lead to that cell, as a string.
            # e.g. "HV" means that horizontal and vertical movements were the
            # best.
            backtrack_matrix.set_value(i, j, "".join(check_argmax([s1_gap, s2_gap, mut])))
    
    # If semiglobal alignment, get the best value on the last row.
    align_score = max(S[len(s1), :]) if semiglobal else S[len(s1), len(s2)]
    return [align_score, S, backtrack_matrix]
        
##########################################
# TEST ###################################
##########################################


gap_penalty = -1

start_time = timeit.default_timer()
# Load the sequences and test their edit distance
for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
    for j, seq_record_j in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
        if i > j:
            #            print("Comparing:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
            #            print("\t", seq_record_j.id, "-- length:", len(seq_record_j))
            [score, edit_matrix, backtrack_matrix] = global_aligner_2(
                seq_record_i.seq,  seq_record_j.seq, gap_penalty=-1, matrix=MatrixInfo.blosum62)
            align_list = backtrack_sequence_rec(
                seq_record_i.seq, seq_record_j.seq, backtrack_matrix, k=1)

#            for p in align_list:
#                print(str(p) + "\n\n")
#            print("DONE")
#            print("MY ALIGNER:", score)
#            print("BIOPYTHON ALIGNER", pairwise2.align.globaldx(seq_record_i.seq, seq_record_j.seq, MatrixInfo.blosum62, score_only = True))
#            print("\n")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")


s1 = "THISLINE"
s2 = "ISALIGNED"
[score, edit_matrix, backtrack_matrix] = global_aligner_2(s1, s2, gap_penalty=-8, matrix=MatrixInfo.blosum62, semiglobal=True)
# print(edit_matrix)
print(backtrack_matrix)

edit_frame = pd.DataFrame(edit_matrix)
edit_frame.index = list(" " + s1)
edit_frame.columns = list(" " + s2)

align_list = semiglobal_backtrack(s1, s2, edit_matrix, backtrack_matrix)

for p in align_list:
    print(str(p) + "\n")
print("DONE")


[score_2, edit_matrix_2, backtrack_matrix_2] = global_aligner_affine_penalty(s1, s2, gap_penalty=-2, gap_opening_penalty=-3, matrix=MatrixInfo.blosum62, semiglobal=True)

# print(edit_matrix)                            
print(backtrack_matrix_2)

edit_frame_2 = pd.DataFrame(edit_matrix_2)
edit_frame_2.index = list(" " + s1)
edit_frame_2.columns = list(" " + s2)

align_list_2 = semiglobal_backtrack(s1, s2, edit_matrix_2, backtrack_matrix_2)

for p in align_list_2:
    print(str(p) + "\n")
print("DONE")
