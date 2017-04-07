import gor_process_input as gori
import pandas as pd
import numpy as np
import timeit
import pickle

class SecStructPrediction:
    def __init__(self, pdb_code="", prediction="", real_sec_structure="", q3=0, mcc=0, overall_prediction=""):
        self.pdb_code = pdb_code
        self.prediction = prediction
        self.real_sec_structure = real_sec_structure
        self.q3 = q3
        self.mcc = mcc
        self.overall_prediction = overall_prediction

    def __str__(self):
        return self.pdb_code + ":\n" +\
               self.real_sec_structure + "\n" +\
               self.prediction +\
               "\nMETRICS -- Q3:" + "{:.3f}".format(self.q3) + " -- MCC:" + "{:.3f}".format(self.mcc) + " -- TOT. PRED.:" + self.overall_prediction + "\n"

    def __repr__(self):
        return str(self)

def predict_sec_structures(input_data, data_type="dssp", print_details=False):

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
    for p_i, p in enumerate(list(set(input_data.PDB_code))):
        # Measure the accuracy over a single protein.
        acc = 0
        pred = ""
        # Extract aminoacid and secondary structures for a given protein.
        a_list = input_data.loc[input_data['PDB_code'] == p].a
        s_list = input_data.loc[input_data['PDB_code'] == p].s
        for i in range(0, len(a_list)):
            # Predict the secondary structure for a given aminoacid of the protein.
            res = gori.compute_gor_3(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ, l_1_out=True)
            # Add the new secondary structure that is predicted
            pred += res[0]
            if res[0] == s_list.iat[i]:
                acc += 1

        prediction = SecStructPrediction(p, pred, s_list.to_string(header=False, index=False).replace("\n", ""), acc / len(a_list))
        prediction_list.append(prediction)
        tot_count += len(a_list)
        tot_acc += acc
        # Print the current prediction and the cumulated metrics.
        if print_details:
            print(prediction)
            print("(", p_i + 1,"/", len(list(set(input_data.PDB_code))), ") -- TOTAL ACCURACY:", "{:.3f}".format(tot_acc / tot_count))

    return prediction_list

if __name__ == '__main__':

     # Type of the data to read ("stride", "dssp")
    data_type = "dssp"
    # File name
    file_name = "../data/" + data_type + "_info.txt"
    # Read the data
    input_data = gori.preprocess_input(file_name, gori.aminoacid_codes)

    prediction_list = predict_sec_structures(input_data, data_type, True)




