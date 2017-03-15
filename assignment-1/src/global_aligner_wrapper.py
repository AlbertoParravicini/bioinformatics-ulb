from Bio import SeqIO, SeqRecord
from Bio.SubsMat import MatrixInfo
import utils
import global_aligner_2 as gb2
import global_aligner as gb1

import timeit


def global_aligner(seq_1, seq_2, gap_penalty: object = -1, gap_opening_penalty=-10, num_of_alignments=1,
                   edit_function: object = utils.sub_matrices_distance,
                   matrix: object = MatrixInfo.pam120,
                   semiglobal: object = False):
    s1 = seq_1.seq if isinstance(seq_1, SeqRecord.SeqRecord) else seq_1
    s2 = seq_2.seq if isinstance(seq_2, SeqRecord.SeqRecord) else seq_2

    if gap_opening_penalty is not 0:
        # Use the aligner with affine penalty.
        [score, score_matrix, backtrack_matrix] = gb2.global_aligner_affine_penalty_2(
            s1, s2, gap_penalty, gap_opening_penalty, matrix=matrix, semiglobal=semiglobal)
    else:
        if num_of_alignments == 1:
            # If we require just one optimal alignment,
            # use the basic global aligner, without having to generate a backtrack matrix.
            [score, score_matrix] = gb1.global_aligner(s1, s2, gap_penalty, matrix=matrix, semiglobal=semiglobal)
        else:
            # Use the default global aligner.
            [score, score_matrix, backtrack_matrix] = gb2.global_aligner_2(s1, s2, gap_penalty, matrix=matrix, semiglobal=semiglobal)


    if num_of_alignments == 1 and gap_opening_penalty == 0:
        [aligned_s1, aligned_s2, match_sequence] = gb1.backtrack_matrix(s1, s2, score_matrix, gap_penalty, edit_function, matrix, semiglobal)
        align_list = [utils.Alignment(aligned_s1, aligned_s2, match_sequence)]
    else:
        # Use the backtrack matrix to find the alignments.
        if semiglobal:
            align_list = gb2.semiglobal_backtrack(s1, s2, score_matrix, backtrack_matrix, num_of_alignments)
        else:
            align_list = gb2.backtrack_sequence_rec(s1, s2, backtrack_matrix, num_of_alignments)

    # Set the score of the alignment
    for align in align_list:
        align.score = score
    return align_list


start_time = timeit.default_timer()
# Load the sequences and test their edit distance
for i, seq_record_i in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
    for j, seq_record_j in enumerate(SeqIO.parse("../data/WW-sequence.fasta", "fasta")):
        if i > j:
            print("Comparing:\n\t", seq_record_i.id, "-- length:", len(seq_record_i))
            print("\t", seq_record_j.id, "-- length:", len(seq_record_j))
            align_list = global_aligner(seq_record_i,  seq_record_j, -1, 0, 1, matrix=MatrixInfo.blosum62)

            for p in align_list:
              print(str(p) + "\n\n")
end_time = timeit.default_timer()
print("! -> EXECUTION TIME:", (end_time - start_time), "\n")
