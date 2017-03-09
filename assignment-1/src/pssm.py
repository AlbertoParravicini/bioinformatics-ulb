import numpy as np
from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo
import utils
import pandas as pd
from itertools import compress

import local_aligner
import global_aligner_2 as gb2
import timeit

filename = "../data/msa-results-omega.fasta"

##########################################
# PSSM ###################################
##########################################

def build_pssm(records_filename, records_format="fasta", alpha=None, beta=None, gap_penalty_coeff=-6, gap_penalty_vector=None):
    """
    Build the Position specific scoring matrix for a given set of aligned sequences.

    Parameters
    ----------
	records_filename: str
		The name of the file that contains the sequences to be loaded.

	records_format: str, optional
		Format in which the sequences are stored (for instance "fasta").
		For the full list, refer to the documentation of BioPython.SeqIO.

	alpha: float, optional
		Parameter used to build the PSSM.
		By default, it is N_seq - 1

	beta: float, optional
		Parameter used to build the PSSM.
		By default, it is sqrt(N_seq)

    gap_penalty_coeff: float, optional
        Maximum value that will be assigned as gap penalty, if no gap_penalty_vector is provided.
        The gap penalties will be inversely proportional to the frequencies of gaps in the aligned sequences.

    gap_penalty_vector: array, optional
        Array with length equal to each aligned sequences, it contains the gap penalities to apply to each position.

	Returns
    ----------
    pssm: pandas.DataFrame
        The PSSM of the record set.

    frequency_matrix: pandas.DataFrame
        The frequency matrix of the record set.

    consensus: str
        The consensus of the given record set.
        For each position, it gives the most frequent aminoacid.
    """
    records = list(SeqIO.parse(records_filename, records_format))

    # Parameters
    n_seq = len(records)
    seq_length = len(records[0].seq)
    # Set the default values if needed.
    if alpha is None:
        alpha = n_seq - 1
    if beta is None:
        beta = np.sqrt(n_seq)

    # Matrices:

    # Matrix that contains the frequencies of each aminoacid in each position of the sequence.
    # The frequency of the gap character is counted too.
    frequence_matrix = pd.DataFrame(0, index=list(utils.aminoacid_list + "-"), columns=range(seq_length))
    # Intermediate matrix that is used in the computation.
    q_matrix = pd.DataFrame(0, index=list(utils.aminoacid_list), columns=range(seq_length))
    # PSSM matrix, with the gap character row added at the end.
    pssm = pd.DataFrame(0, index=list(utils.aminoacid_list + "-"), columns=range(seq_length))

    # Build frequency matrix
    for i in range(len(frequence_matrix.index)):
        for j in range(len(frequence_matrix.columns)):
            frequence_matrix.iloc[i, j] = np.sum([frequence_matrix.index[i] == seq_n.seq[j] for seq_n in records])
    frequence_matrix /= n_seq

    # Build intermediate matrix
    for i in range(len(q_matrix.index)):
        q_matrix.iloc[i, :] = alpha * frequence_matrix.iloc[i, :] + beta * utils.aminoacid_prob[q_matrix.index[i]]
    q_matrix /= (alpha + beta)

    # Build pssm
    for i in range(len(pssm.index)-1):
        pssm.iloc[i, :] = np.log(q_matrix.iloc[i, :] / utils.aminoacid_prob[pssm.index[i]])

    # Build the PSSM row relative to the gap character.
    # Use the provided vector or build it using the frequencies of the gap character.
    if gap_penalty_vector is None:
        pssm.loc["-", :] = np.ceil(gap_penalty_coeff * (-np.tanh(frequence_matrix.loc["-", :]) + 1))
    else:
        pssm.loc["-", :] = gap_penalty_vector

    return [pssm, frequence_matrix, find_consensus(frequence_matrix)]



##########################################
# FIND CONSENSUS #########################
##########################################

def find_consensus(frequency_matrix):
    return frequency_matrix.apply(np.argmax, axis=0, reduce=True).tolist()



##########################################
# PSSM ALIGNER ###########################
##########################################

def pssm_aligner(seq, pssm, k=1, sub_alignments_num=1, consensus=None):
    """
    Align a given sequence to a given pssm, with a local aligner.

    Parameters
    ----------
    seq: string
        The input aminoacid sequences to be aligned.

    pssm: pandas.DataFrame
        The PSSM against which the input sequence is aligned.

    k: int, optional
        The number of best alignment to return, in case there are more alignments with the same optimal score.

    sub_alignments_num: int, optional
        The number of novel subalignments to be produced.

    consensus: str, optional
        The consensus sequence of the PSSM.
        If available, the input sequence will be visually aligned to it,
        to provide a better understanding of the alignment

    Returns
    ----------
    array of Alignment
        A list of k alignments, of class utils.Alignment
    """
    alignments = []

    # Build the initial score matrix.
    [score, S, backtrack_matrix, i_max, j_max] = pssm_aligner_score(seq, pssm)
    for n in range(sub_alignments_num):
        # If the consensus isn't given, the string is visually aligned to the column numbers.
        # In short, only the alignment of the given sequence, and the score will be of any use.
        seq2 = [str(i) for i in pssm.columns.values][1:] if consensus is None else consensus
        align_list_n = gb2.backtrack_sequence_rec(seq[:i_max], seq2[:j_max], backtrack_matrix.iloc[:i_max + 1, :j_max + 1], k=k)

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
            coordinate_list = reconstruct_sequence(seq, pssm, S, backtrack_matrix.iloc[:i_max + 1, :j_max + 1])
            update_score_matrix(seq, pssm, S, coordinate_list, backtrack_matrix)

            # Find the new maximum value in the matrix.
            [i_max, j_max] = np.unravel_index(np.argmax(S), S.shape)
            score = S[i_max, j_max]
            if i_max == 0 and j_max == 0:
                break

    return alignments



##########################################
# PSSM ALIGNER SCORE #####################
##########################################

def pssm_aligner_score(seq, pssm):
    n_row = len(seq) + 1
    n_col = len(pssm.columns) + 1
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
            # Compute the possible movements, and then keep the best.
            seq_gap = S[i-1, j] + pssm.loc["-", j-1]
            pssm_gap = S[i, j-1] + (0 if j == 1 else pssm.loc["-", j-2])
            mut = S[i-1, j-1] + pssm.loc[seq[i-1], j-1]
            # In the local aligner, don't accept negative scores!
            S[i, j] = max(seq_gap, pssm_gap, mut, 0)

            if S[i, j] >= score_max:
                score_max = S[i, j]
                i_max = i
                j_max = j
            # Write in the matrix the movement that lead to that cell, as a string.
            # e.g. "HV" means that horizontal and vertical movements were the
            # best.
            # In local alignment, "X" means that 0 was the maximum value, and all the movements gave a negative score.
            # The backtracking will stop when an "X" is encountered.
            backtrack_matrix.set_value(i, j, "".join(local_aligner.check_argmax([seq_gap, pssm_gap, mut, 0])))

    return [score_max, S, backtrack_matrix, i_max, j_max]



##########################################
# BACKTRACK ##############################
##########################################

def reconstruct_sequence(seq, pssm, S, backtrack_matrix):

    coordinate_list = []
    [i, j] = backtrack_matrix.shape
    i-=1
    j-=1

    while i > 0 or j > 0:
        val = S[i, j]
        # If a 0 is found, interrupt the traceback
        if val == 0:
            coordinate_list.append([i, j])
            break

        # Consider 0 to handle the first row/column
        if j == 0:
            print("ERORRE J=0")
        seq_gap = 0 if i == 0 else S[i-1, j] + pssm.loc["-", j-1]
        pssm_gap = 0 if j == 0 else S[i, j-1] + (0 if j == 1 else pssm.loc["-", j-2])
        mut = S[i-1, j-1] + pssm.loc[seq[i-1], j-1]

        # Append the current location to the coordinate list.
        coordinate_list.append([i, j])

        # Match s1 to a gap, move vertically
        if i > 0 and val == seq_gap:
            i -= 1
        # Match s2 to a gap, move horizontally
        elif j > 0 and val == pssm_gap:
            j -= 1
        # Substitution, diagonal movement
        elif i > 0 and j > 0 and val == mut:
            i -= 1
            j -= 1
        else:
            raise ValueError("val={0}, but we have seq_gap={1}, pssm_gap={2}, mut={3}".format(val, seq_gap, pssm_gap, mut))

    coordinate_list.reverse()
    return coordinate_list



##########################################
# UPDATE SCORE MATRIX ####################
##########################################

def update_score_matrix(seq, pssm, S, coordinate_list, backtrack_matrix):

    for [i, j] in coordinate_list:
        # Set the current value to 0 in the optimal matching sequence.
        S[i, j] = 0
        backtrack_matrix.iloc[i, j] = "X"
        # Recompute the values below S[i, j]
        if j > 0 and i > 0:
            # Update the values below the current position.
            for i_i in range(i+1, len(seq)+1):

                seq_gap = 0 if i_i == 0 else S[i_i-1, j] + pssm.loc["-", j-1]
                pssm_gap = 0 if j == 0 else S[i_i, j-1] + (0 if j == 1 else pssm.loc["-", j-2])
                mut = S[i_i-1, j-1] + pssm.loc[seq[i_i-1], j-1]
                # In the local aligner, don't accept negative scores!
                if backtrack_matrix.iloc[i_i, j] != "X":
                    S[i_i, j] = max(seq_gap, pssm_gap, mut, 0)
            # Update the values on the right the current position.
            for j_j in range(j+1, len(pssm.columns)+1):

                seq_gap = 0 if i == 0 else S[i-1, j_j] + pssm.loc["-", j_j-1]
                pssm_gap = 0 if j_j == 0 else S[i, j_j-1] + (0 if j_j == 1 else pssm.loc["-", j_j-2])
                mut = S[i-1, j_j-1] + pssm.loc[seq[i-1], j_j-1]
                # In the local aligner, don't accept negative scores!
                if backtrack_matrix.iloc[i, j_j] != "X":
                    S[i, j_j] = max(seq_gap, pssm_gap, mut, 0)

    for i in range(coordinate_list[-1][0]+1, len(seq)+1):
        for j in range(coordinate_list[-1][1]+1, len(pssm.columns)+1):
            seq_gap = S[i-1, j] + pssm.loc["-", j-1]
            pssm_gap = S[i, j-1] + (0 if j == 1 else pssm.loc["-", j-2])
            mut = S[i-1, j-1] + pssm.loc[seq[i-1], j-1]
            if backtrack_matrix.iloc[i, j] != "X":
                S[i, j] = max(seq_gap, pssm_gap, mut, 0)



##########################################
# TEST ###################################
##########################################

[pssm, freq_matrix, consensus] = build_pssm(filename,gap_penalty_coeff=-1)

start_time = timeit.default_timer()
# Load the sequences and test their edit distance
for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
    print("Aligning:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
    align_list = pssm_aligner(seq_record_i.seq, pssm, consensus=consensus)
    for p in align_list:
        print(str(p) + "\n\n------------------------------------------------------")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")

# start_time = timeit.default_timer()
# # Load the sequences and test their edit distance
# for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
#     #print("Aligning:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
#     align_list = pssm_aligner(seq_record_i.seq, pssm, consensus=None)
#     for p in align_list:
#         print(str(p.s1))
# end_time = timeit.default_timer()
# print("! -> EXECUTION TIME:", (end_time - start_time), "\n")

#
# start_time = timeit.default_timer()
# # Load the sequences and test their edit distance
# for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
#     align_list = pssm_aligner(seq_record_i.seq, pssm, consensus=None)
#     for p in align_list:
#         aligner = p.s2.replace("-", "")
#         aligned_string = " " * (int(aligner[0])-1) + p.s1
#         print(aligned_string)
# end_time = timeit.default_timer()
# print("! -> EXECUTION TIME:", (end_time - start_time), "\n")
