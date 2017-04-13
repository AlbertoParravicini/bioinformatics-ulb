import gor_process_input as gori
import pandas as pd
import numpy as np
import timeit
import pickle

class SecStructPrediction:
    """
    Class used to store the secondary structure predictions of proteins.
    It can store the PDB code of a protein, its aminoacid sequence, the real secondary structure and the predicted one,
    and more. Fields can be left empty if the information are not available.
    """
    def __init__(self, pdb_code="", protein="", prediction="", real_sec_structure="", q3=0, mcc_h=0, mcc_b=0, mcc_c=0, mcc=0, overall_prediction=""):
        self.pdb_code = pdb_code
        self.protein = protein
        self.prediction = prediction
        self.real_sec_structure = real_sec_structure
        self.q3 = q3
        self.mcc_h = mcc_h
        self.mcc_b = mcc_b
        self.mcc_c = mcc_c
        self.mcc = mcc
        self.overall_prediction = overall_prediction

    def __str__(self):
        return self.pdb_code + ":\n" +\
               self.protein + "\n" +\
               self.real_sec_structure + "\n" +\
               self.prediction +\
               "\nMETRICS -- Q3:" + "{:.3f}".format(self.q3) +\
               " -- MCC_h:" + "{:.3f}".format(self.mcc_h) +\
               " -- MCC_b:" + "{:.3f}".format(self.mcc_b) +\
               " -- MCC_c:" + "{:.3f}".format(self.mcc_c) +\
               " -- MCC:" + "{:.3f}".format(self.mcc) +\
               " -- TOT. PRED.:" + self.overall_prediction + "\n"

    def __repr__(self):
        return str(self)


#%% GOR

def compute_gor_3(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ, l_1_out=False, l1o_dict=None, l1o_sjaj=None):
    """
    Compute the GOR III information value for the given aminoacid, and return the best secondary structure prediction
    :param a_list: pandas.Series; input aminoacid sequence.
    :param i: position in the sequence for which the prediction is done.
    :param sj_aj_ajm_dict: dictionary used by the GOR III model.
    :param sj_aj_matrix: occurrency matrix used by the GOR III model.
    :param a_occ: occurrencies of each aminoacid, used by the GOR III model.
    :param l_1_out: boolean; do leave one out on the given GOR III model.
    :param l1o_dict:
    :param l1o_sjaj:
    :return: a tuple containing (structure_prediction, prediction_score)
    """
    if i < 0 or i >= len(a_list):
        print("Index out of range:", i, "-- List length:", len(a_list))
        raise IndexError

    min_range = max(-i, -8)
    max_range = min(9, len(a_list) - i)

    a_j = a_list.iat[i]


    info_tot = {"h": 0, "b": 0, "c": 0}
    for s in info_tot.keys():
        for m in range(min_range, max_range):
            a_jm = a_list.iat[i + m]

            # Values to subtract in "leave-one-out"
            # sub_1 = (l1o_dict[s].at[a_j, a_jm, m] if l_1_out else 0)
            # sub_2 = (l1o_sjaj.at[s, a_j] if l_1_out else 0)
            sub_1 = (1 if l_1_out else 0)
            sub_2 = (1 if l_1_out else 0)

            # Num of times s_j appears with a_j, and a_jm is lag position distant.
            num_1 = sj_aj_ajm_dict[s].at[a_j, a_jm, m] - sub_1
            # Num of times a_j appears with a secondary structure different from s_j,
            # and a_jm is lag position distant.
            den_1 = sum([sj_aj_ajm_dict[s_i].at[a_j, a_jm, m] for s_i in info_tot.keys()]) - num_1 - sub_1

            # Number of times aminoacid a_j appears together with structure s_j.
            den_2 = sj_aj_matrix.at[s, a_j] - sub_2
            # Number of times aminoacid a_j appears with a different structure from s_j.
            num_2 = a_occ.at[a_j] - den_2 - sub_2

            # Add up the information value.
            info_tot[s] += np.log(num_1) - np.log(den_1) + np.log(num_2) - np.log(den_2)

    # Predict the secondary structure with highest value.
    return [max(info_tot, key=info_tot.get), max(info_tot.values())]






def predict_sec_structures(input_data, data_type="dssp", print_details=False):
    """
    Predict the secondary structures of the input data set.
    The input data set is divided in proteins, and for each protein are given
    the secondary structure prediction, and accuracy metrics.
    The results are stored in a file.
    :param input_data: pandas.DataFrame, the input sequence of proteins.
    :param data_type: string, the source of the input data, "dssp" or "stride"
    :param print_details: boolean, print the details of the prediction.
    :return: list of SecStructPrediction.
    """

    # Load data
    dict_file_name = "sj_aj_ajm_dict_"+ data_type + ".p"

    file_object = open(dict_file_name.encode('utf-8').strip(), 'rb')
    sj_aj_ajm_dict = pickle.load(file_object)

    # Compute sj_aj_matrix abd a_occ;
    # it's done again as it's fast, no need to store the precomputed matrices.
    a_occ = input_data["a"].value_counts()
    sj_aj_matrix = gori.build_sj_aj_matrix(input_data.s, input_data.a)

    # List that contains all the predictions, divided by protein.
    prediction_list = []
    # Measure the overall accuracy.
    tot_acc = 0
    tot_count = 0
    for p_i, p in enumerate(list(set(input_data.PDB_code ))):
        # Measure the accuracy over a single protein.
        acc = 0
        pred = ""
        # Extract aminoacid and secondary structures for a given protein.
        a_list = input_data.loc[input_data['PDB_code'] == p].a
        s_list = input_data.loc[input_data['PDB_code'] == p].s


        # Build a temporary dictionary for the protein that is examined,
        # it will be used for "leave-one-out"
        temp_dict = None
        temp_sjaj = None
        # temp_dict = gori.build_sj_aj_ajm(pd.DataFrame({"PDB_code_and_chain": p, "a": a_list, "s": s_list}), print_details=False)
        # temp_sjaj = gori.build_sj_aj_matrix(s_list, a_list)
        for i in range(0, len(a_list)):
            # Predict the secondary structure for a given aminoacid of the protein.
            res = compute_gor_3(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ, l_1_out=True, l1o_dict=temp_dict, l1o_sjaj=temp_sjaj)
            # Add the new secondary structure that is predicted
            pred += res[0]
            if res[0] == s_list.iat[i]:
                acc += 1

        mcc_h=compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "h")
        mcc_b=compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "b")
        mcc_c=compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "c")

        prediction = SecStructPrediction(pdb_code=p,
                                         protein=a_list.to_string(header=False, index=False).replace("\n", ""),
                                         prediction=pred,
                                         real_sec_structure=s_list.to_string(header=False, index=False).replace("\n", ""),
                                         q3=acc / len(a_list),
                                         mcc_h=mcc_h,
                                         mcc_b=mcc_b,
                                         mcc_c=mcc_c,
                                         mcc=(mcc_h + mcc_b + mcc_c)/3
                                         )

        prediction_list.append(prediction)
        tot_count += len(a_list)
        tot_acc += acc
        # Print the current prediction and the cumulated metrics.
        if print_details:
            print(prediction)
            print("(", p_i + 1,"/", len(list(set(input_data.PDB_code))), ") -- TOTAL ACCURACY:", "{:.3f}".format(tot_acc / tot_count))

    return prediction_list

def compute_mcc(prediction_seq, real_seq, sec_structure):
    """
    Compute the MCC for a given secondary sequence prediction.
    :param prediction_seq: string, the sequence of predicted secondary structures.
    :param real_seq: string, the sequence of real secondary structures.
    :param sec_structure: string, the secondary structure to use as target (usually one of "h", "b", "c")
    :return: double
    """
    tp = sum([prediction_seq[i] == sec_structure and real_seq[i] == sec_structure for i in range(len(prediction_seq))])
    fn = sum([prediction_seq[i] != sec_structure and real_seq[i] == sec_structure for i in range(len(prediction_seq))])
    fp = sum([prediction_seq[i] == sec_structure and real_seq[i] != sec_structure for i in range(len(prediction_seq))])
    tn = sum([prediction_seq[i] != sec_structure and real_seq[i] != sec_structure for i in range(len(prediction_seq))])
    # If all the values at denominator are different from 0, use the standard formula;
    if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) != 0:
        mcc = (tp*tn - fp*fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    else:
        # If one value is equal to 0, set the denominator to 1.
        # See https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        # Doing so will give an mcc af 0, i.e. a null correlation.
        mcc = tp*tn - fp*fn
    return mcc

if __name__ == '__main__':

    # Type of the data to read ("stride", "dssp")
    data_type = "stride"
    # File name
    file_name = "../data/" + data_type + "_info.txt"
    # Read the data
    input_data = gori.preprocess_input(file_name, gori.aminoacid_codes)

    # Append the PDB chain code to the PDB code, to obtain unique protein identifier.
    input_data.PDB_code = input_data.PDB_code + "_" + input_data.PDB_chain_code

    # Make the predictions.
    prediction_list = predict_sec_structures(input_data, data_type, True)
    # Overall Q3
    q3_tot = np.mean([x.q3 for x in prediction_list])
    print("\nOVERALL Q3:", q3_tot)

    # Load CATH file
    cath = pd.read_csv("../data/cath_info.txt", header=None, sep="\t",
                         names=["PDB_code", "PDB_chain", "overall_structure"])
     
    # Add a composite key to the cath data
    cath["PDB_code_and_chain"] = cath.PDB_code + "_" + cath.PDB_chain


    # Save predictions in a file.
    result_frame = pd.DataFrame(index=range(len(prediction_list)), columns=["PDB_code_and_chain",
                                                                            "length",
                                                                            "prediction",
                                                                            "real_sec_structure",
                                                                            "num_predicted_h",
                                                                            "num_predicted_b",
                                                                            "num_predicted_c",
                                                                            "num_real_h",
                                                                            "num_real_b",
                                                                            "num_real_c",
                                                                            "num_am_a",
                                                                            "num_am_r",
                                                                            "num_am_n",
                                                                            "num_am_d",
                                                                            "num_am_c",
                                                                            "num_am_q",
                                                                            "num_am_e",
                                                                            "num_am_g",
                                                                            "num_am_h",
                                                                            "num_am_i",
                                                                            "num_am_l",
                                                                            "num_am_k",
                                                                            "num_am_m",
                                                                            "num_am_f",
                                                                            "num_am_p",
                                                                            "num_am_s",
                                                                            "num_am_t",
                                                                            "num_am_w",
                                                                            "num_am_y",
                                                                            "num_am_v",
                                                                            "protein",
                                                                            "q3",
                                                                            "mcc_h",
                                                                            "mcc_b",
                                                                            "mcc_c",
                                                                            "mcc",
                                                                            "overall_pred",
                                                                            "overall_structure_real"])
    for p_i, p in enumerate(prediction_list):
        result_frame.iloc[p_i, 0] = p.pdb_code
        result_frame.iloc[p_i, 1] = len(p.prediction)
        result_frame.iloc[p_i, 2] = p.prediction
        result_frame.iloc[p_i, 3] = p.real_sec_structure
        result_frame.iloc[p_i, 4] = p.prediction.count("h")
        result_frame.iloc[p_i, 5] = p.prediction.count("b")
        result_frame.iloc[p_i, 6] = p.prediction.count("c")
        result_frame.iloc[p_i, 7] = p.real_sec_structure.count("h")
        result_frame.iloc[p_i, 8] = p.real_sec_structure.count("b")
        result_frame.iloc[p_i, 9] = p.real_sec_structure.count("c")
        result_frame.iloc[p_i, 10] = p.protein.count("a")
        result_frame.iloc[p_i, 11] = p.protein.count("r")
        result_frame.iloc[p_i, 12] = p.protein.count("n")
        result_frame.iloc[p_i, 13] = p.protein.count("d")
        result_frame.iloc[p_i, 14] = p.protein.count("c")
        result_frame.iloc[p_i, 15] = p.protein.count("q")
        result_frame.iloc[p_i, 16] = p.protein.count("e")
        result_frame.iloc[p_i, 17] = p.protein.count("g")
        result_frame.iloc[p_i, 18] = p.protein.count("h")
        result_frame.iloc[p_i, 19] = p.protein.count("i")
        result_frame.iloc[p_i, 20] = p.protein.count("l")
        result_frame.iloc[p_i, 21] = p.protein.count("k")
        result_frame.iloc[p_i, 22] = p.protein.count("m")
        result_frame.iloc[p_i, 23] = p.protein.count("f")
        result_frame.iloc[p_i, 24] = p.protein.count("p")
        result_frame.iloc[p_i, 25] = p.protein.count("s")
        result_frame.iloc[p_i, 26] = p.protein.count("t")
        result_frame.iloc[p_i, 27] = p.protein.count("w")
        result_frame.iloc[p_i, 28] = p.protein.count("y")
        result_frame.iloc[p_i, 29] = p.protein.count("v")
        result_frame.iloc[p_i, 30] = p.protein
        result_frame.iloc[p_i, 31] = p.q3
        result_frame.iloc[p_i, 32] = p.mcc_h
        result_frame.iloc[p_i, 33] = p.mcc_b
        result_frame.iloc[p_i, 34] = p.mcc_c
        result_frame.iloc[p_i, 35] = p.mcc
        result_frame.iloc[p_i, 36] = p.overall_prediction
        result_frame.iloc[p_i, 37] = cath.loc[cath['PDB_code_and_chain'] == p.pdb_code].overall_structure.iloc[0]
        
    result_frame.to_csv("../data/pred_result_" + data_type + ".csv", index=False)



    # OVERALL ACCURACY DSSP: 0.631
    # OVERALL ACCURACY STRIDE: 0.636








