import pandas as pd
import numpy as np
import gor_compute_predictions as gorp
import gor_process_input as gori
import pickle

#%%
def compute_gor_5(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ):
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
    
    if a_j != "-":
        for s in info_tot.keys():
            for m in range(min_range, max_range):
                a_jm = a_list.iat[i + m]
                
                if a_jm != "-":   
                    # Num of times s_j appears with a_j, and a_jm is lag position distant.
                    num_1 = sj_aj_ajm_dict[s].at[a_j, a_jm, m] 
                    # Num of times a_j appears with a secondary structure different from s_j,
                    # and a_jm is lag position distant.
                    den_1 = sum([sj_aj_ajm_dict[s_i].at[a_j, a_jm, m] for s_i in info_tot.keys()]) - num_1 
        
                    # Number of times aminoacid a_j appears together with structure s_j.
                    den_2 = sj_aj_matrix.at[s, a_j] 
                    # Number of times aminoacid a_j appears with a different structure from s_j.
                    num_2 = a_occ.at[a_j] - den_2 
        
                    # Add up the information value.
                    info_tot[s] += np.log(num_1) - np.log(den_1) + np.log(num_2) - np.log(den_2)
    
        # Predict the secondary structure with highest value.
        return [max(info_tot, key=info_tot.get), max(info_tot.values())]
    else:
        return ["-", 1]


def predict_sec_structures_gor_5(input_data, data_type="dssp", print_details=1):
    """
    Predict the secondary structures of the input data set.
    The input data set is divided in proteins, and for each protein are given
    the secondary structure prediction, and accuracy metrics.
    The results are stored in a file.
    :param input_data: pandas.DataFrame, the input sequence of proteins.
    :param data_type: string, the source of the input data, "dssp" or "stride"
    :param print_details: int, print the details of the prediction (higher == more details).
    :return: list of SecStructPrediction.
    """

    # Load data
    dict_file_name = "sj_aj_ajm_dict_"+ data_type + ".p"

    file_object = open(dict_file_name.encode('utf-8').strip(), 'rb')
    sj_aj_ajm_dict = pickle.load(file_object)
    file_object.close()
    
    # Load data
    a_occ_file_name = "a_occ_"+ data_type + ".p"
    file_object = open(a_occ_file_name.encode('utf-8').strip(), 'rb')
    a_occ = pickle.load(file_object)
    file_object.close()
    
    # Load data
    mat_file_name = "sj_aj_matrix_"+ data_type + ".p"

    file_object = open(mat_file_name.encode('utf-8').strip(), 'rb')
    sj_aj_matrix = pickle.load(file_object)
    file_object.close()

    # List that contains all the predictions, divided by protein.
    prediction_list = []
    # Measure the overall accuracy.
    tot_acc = 0
    tot_count = 0
    try:
        for p_i, p in enumerate(list(set(input_data.PDB_code))):
            # Measure the accuracy over a single protein.
            acc = 0
            pred = ""
            # Extract aminoacid and secondary structures for a given protein.
            a_list = input_data.loc[input_data['PDB_code'] == p].a
    #        s_list = input_data.loc[input_data['PDB_code'] == p].s
            
            for i in range(0, len(a_list)):
                # Predict the secondary structure for a given aminoacid of the protein.
                if a_list.iat[i] != "-":
                    res = compute_gor_5(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ)
                    # Add the new secondary structure that is predicted
                    pred += res[0]
                else:
                    pred += "-"
    #            if res[0] == s_list.iat[i]:
    #                acc += 1
    
    #        mcc_h, tp_h, _, _, _ = compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "h")
    #        mcc_b, tp_b, _, _, _ = compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "b")
    #        mcc_c, tp_c, _, _, _ = compute_mcc(pred, s_list.to_string(header=False, index=False).replace("\n", ""), "c")
    #        
            
    #        h_count = s_list.to_string().count("h")
    #        b_count = s_list.to_string().count("b")
    #        c_count = s_list.to_string().count("c")
            prediction = gorp.SecStructPrediction(pdb_code=p,
                                             protein=a_list.to_string(header=False, index=False).replace("\n", ""),
                                             prediction=pred,
    #                                         real_sec_structure=s_list.to_string(header=False, index=False).replace("\n", ""),
    #                                         q3= acc / len(a_list),
    #                                         q3_h = (tp_h / h_count) if h_count > 0 else 1,
    #                                         q3_b = (tp_b / b_count) if b_count > 0 else 1,
    #                                         q3_c = (tp_c / c_count) if c_count > 0 else 1,
    #                                         mcc_h=mcc_h,
    #                                         mcc_b=mcc_b,
    #                                         mcc_c=mcc_c,
    #                                         mcc=(mcc_h + mcc_b + mcc_c)/3
                                             )
    
            prediction_list.append(prediction)
            tot_count += len(a_list)
            tot_acc += acc
            # Print the current prediction and the cumulated metrics.
            if print_details > 1:
                print(prediction)
            if print_details > 0:
                print("(", p_i + 1,"/", len(list(set(input_data.PDB_code))), ")")
    except KeyError:
        print("PRED:", pred)
        raise
    return prediction_list

#%%

def gor_5_overall_structure(reference_protein, prediction_list):
    """
    Predict the secondary structures of the reference protein,
    by considering a list of predictions done on proteins that have been aligned to the reference protein.
    :param reference_protein: string, the name of the protein for which the secondary structures are computed;
        the reference_protein must be included in the prediction_list
    :param prediction_list: list of SecStructPrediction. It contains the predictions for the aligned proteins.
    :return: string.
    """
    sec_structs = ["h", "b", "c"]
    pred = ""
    # For each position in the aligned sequence, compute the most common predicted secondary structure.
    for i in range(len(prediction_list[0].prediction)):
        i_th_col = [x.prediction[i] for x in prediction_list]
        dict_am_count = {a: sum(s == a for s in i_th_col) for a in sec_structs}
        pred += max(dict_am_count, key=dict_am_count.get)
    
    reference_protein_seq = None
    # Find the SecStructPrediction relative to the reference_protein.
    for p in prediction_list:
        if p.pdb_code == reference_protein:
            reference_protein_seq = p
            break

    final_prediction = ""
    # Add a predicted secondary structure only if it doesn't correspond to a gap
    # in the reference_protein (considering its aligned aminoacid sequence).
    for i in range(len(reference_protein_seq.prediction)):
        if reference_protein_seq.protein[i] != "-":
            final_prediction += pred[i]
    return final_prediction

#%%






file_name = "../data/protein_1arl_A_mul.clustal"

# Load the data
input_data = pd.read_csv(file_name,
                         header=None, sep=",",
                         names=["protein_code", "a"])

input_data = input_data.drop(input_data[input_data["protein_code"].isnull()].index).reset_index(drop=True)

protein_set = list(set(input_data.protein_code))

al = input_data.loc[input_data['protein_code'] == input_data.iloc[0,0]].a
seq = al.str.cat(sep="")

aligned_set = pd.DataFrame(index=np.arange(len(protein_set)*len(seq)), columns=["PDB_code", "a"])

for p_i, p in enumerate(protein_set):
    al = input_data.loc[input_data['protein_code'] == p].a
    seq = al.str.cat(sep="")
    for s_i in range(len(seq)):
        aligned_set.iat[p_i*len(seq) + s_i, 0] = p
        aligned_set.iat[p_i*len(seq) + s_i, 1] = seq[s_i]
    
aligned_set.ix[aligned_set.a == "X", 'a'] = "-"
aligned_set.a = aligned_set.a.str.lower()
    
#%%
data_type = "stride"
prediction_list = predict_sec_structures_gor_5(aligned_set, data_type, True)

    
#%%

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
                                                                        "q3_h",
                                                                        "q3_b",
                                                                        "q3_c",
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
    result_frame.iloc[p_i, 32] = p.q3_h
    result_frame.iloc[p_i, 33] = p.q3_b
    result_frame.iloc[p_i, 34] = p.q3_c
    result_frame.iloc[p_i, 35] = p.mcc_h
    result_frame.iloc[p_i, 36] = p.mcc_b
    result_frame.iloc[p_i, 37] = p.mcc_c
    result_frame.iloc[p_i, 38] = p.mcc
    result_frame.iloc[p_i, 39] = p.overall_prediction
    result_frame.iloc[p_i, 40] = ""

result_frame.to_csv("../data/gor_5_pred_result_" + data_type + ".csv", index=False)

#%%
pred = gor_5_overall_structure("1arl_A;", prediction_list)

tot = "CCCCCCCCCCCCCCHHHHHHHHHHHHHHCCCCEEEEEEEECCCCCEEEEEEECCCCCCCC\
EEEEEECCCCCCHHHHHHHHHHHHHHHHHCCCCHHHHHHHHHCEEEEECCCCHHHHHHHH\
HCCCCCCCCCCCCCCCCCCCCCHHHCCCCCCCCCCCECCCCCCCECCCCCCCCHHHHHHH\
HHHHHHCCEEEEEEEEECCCEEEECCCCCCCCCCCHHHHHHHHHHHHHHHHHHHCCCCEE\
EEHHHHCCCCCCCHHHHHHHCCCCEEEEEEECCCCCCHHHCCHHHHHHHHHHHHHHHHHH\
HHHHHHC".lower().replace("e", "b")


pro = "ARSTNTFNYATYHTLDEIYDFMDLLVAEHPQLVSKLQIGRSYEGRPIYVLKFSTGGSNRPAIWIDLGIHSREWIT\
QATGVWFAKKFTEDYGQDPSFTAILDSMDIFLEIVTNPDGFAFTHSQNRLWRKTRSVTSSSLCVGVDANRNWDAG\
FGKAGASSSPCSETYHGKYANSEVEVKSIVDFVKDHGNFKAFLSIHSYSQLLLYPYGYTTQSIPDKTELNQVAKS\
AVAALKSLYGTSYKYGSIITTIYQASGGSIDWSYNQGIKYSFTFELRDTGRYGFLLPASQIIPTAQETWLGVLTI\
MEHTVNN".lower()

tot = list(tot)
pro = list(pro)

for i in range(len(pro)):
    if pro[i] == "x":
        tot[i] = "-"
        pro[i] = "-"

tot = "".join(tot)
pro = "".join(pro)
tot = tot.replace("-", "")
pro = pro.replace("-", "")

acc = 0
for i in range(len(tot)):
    if pred[i] == tot[i]:
        acc+=1
print(acc/len(tot))



#%%
if __name__ == '__main__':
    
    # Open the proteins used for testing.  
    test_proteins_input = pd.read_csv("../data/test_proteins.csv", header=None, sep=",",
                         names=["PDB_code_and_chain","protein","sec_structure","overall_structure"])
    
    # Replace "E" with "B"
    test_proteins_input.sec_structure = test_proteins_input.sec_structure.str.replace("E", "B")
    
    # Build a frame with the appropriate structure
    test_proteins = pd.DataFrame(columns=["PDB_code","a","s"])
    
    for p_i in range(len(test_proteins_input.index)):
        p = test_proteins_input.iloc[p_i,:]
        temp_frame = pd.DataFrame(index=np.arange(len(p.protein)), columns=["PDB_code","a","s"])
        temp_frame["PDB_code"] = p.PDB_code_and_chain
        temp_frame["a"] = list(p.protein)
        temp_frame["s"] = list(p.sec_structure)
        test_proteins = test_proteins.append(temp_frame, ignore_index=True)
     
    # Put to lowercase
    test_proteins.a = test_proteins.a.str.lower()
    test_proteins.s = test_proteins.s.str.lower()
    
    #%%
    
    # Make predictions
    
    data_type = "stride"
    # Make the predictions.
    prediction_list = gorp.predict_sec_structures(test_proteins, data_type, print_details=True, l_1_out=False)
    
    # Overall Q3
    q3_tot = np.mean([x.q3 for x in prediction_list])
    print("\nOVERALL Q3:", q3_tot)

    
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
                                                                            "q3_h",
                                                                            "q3_b",
                                                                            "q3_c",
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
        result_frame.iloc[p_i, 32] = p.q3_h
        result_frame.iloc[p_i, 33] = p.q3_b
        result_frame.iloc[p_i, 34] = p.q3_c
        result_frame.iloc[p_i, 35] = p.mcc_h
        result_frame.iloc[p_i, 36] = p.mcc_b
        result_frame.iloc[p_i, 37] = p.mcc_c
        result_frame.iloc[p_i, 38] = p.mcc
        result_frame.iloc[p_i, 39] = p.overall_prediction
        result_frame.iloc[p_i, 40] = test_proteins_input.loc[test_proteins_input["PDB_code_and_chain"] == p.pdb_code].overall_structure.iloc[0]
    result_frame.to_csv("../data/test_proteins_pred_result_" + data_type + ".csv", index=False)



    #%% Predict with GOR V
    
    test_protein_set = set(test_proteins_input.PDB_code_and_chain)
    #test_protein_set = {"1jsu_C"}
    
    # Used to compute accuracy
    tot_count = 0
    tot_acc = 0
    
    prediction_list = []
    
    for test_p_i, test_p in enumerate(test_protein_set):
        print("WORKING ON:", test_p)
              
        file_name = "../data/protein_" + test_p + "_mul.clustal"

        # Load the data
        input_data = pd.read_csv(file_name,
                                 header=None, sep=",",
                                 names=["protein_code", "a"])
        
        input_data = input_data.drop(input_data[input_data["protein_code"].isnull()].index).reset_index(drop=True)
        
        aligned_protein_set_input = list(set(input_data.protein_code))
        
        al = input_data.loc[input_data['protein_code'] == input_data.iloc[0,0]].a
        seq = al.str.cat(sep="")
        
        aligned_set = pd.DataFrame(index=np.arange(len(aligned_protein_set_input)*len(seq)), columns=["PDB_code", "a"])
        
        for p_i, p in enumerate(aligned_protein_set_input):
            al = input_data.loc[input_data['protein_code'] == p].a
            seq = al.str.cat(sep="")
            for s_i in range(len(seq)):
                aligned_set.iat[p_i*len(seq) + s_i, 0] = p
                aligned_set.iat[p_i*len(seq) + s_i, 1] = seq[s_i]
            
        aligned_set.ix[aligned_set.a == "X", 'a'] = "-"
        aligned_set.a = aligned_set.a.str.lower()
            
        data_type = "stride"
           
        print("BUILDING PREDICTION FOR ALIGNED SEQUENCES...")
        aligned_prediction_list = predict_sec_structures_gor_5(aligned_set, data_type, 1)
        
        print("COMPUTING PREDICTION FOR:", test_p)
        tot_pred = gor_5_overall_structure(test_p, aligned_prediction_list)
        print(test_p, "\n", tot_pred)
        
        a_list = test_proteins[test_proteins["PDB_code"] == test_p].a
        s_list = test_proteins[test_proteins["PDB_code"] == test_p].s
        
        acc = 0
        for i in range(len(tot_pred)):
            if tot_pred[i] == s_list.iloc[i]:
                acc+=1

        mcc_h, tp_h, _, _, _ = gorp.compute_mcc(tot_pred, s_list.to_string(header=False, index=False).replace("\n", ""), "h")
        mcc_b, tp_b, _, _, _ = gorp.compute_mcc(tot_pred, s_list.to_string(header=False, index=False).replace("\n", ""), "b")
        mcc_c, tp_c, _, _, _ = gorp.compute_mcc(tot_pred, s_list.to_string(header=False, index=False).replace("\n", ""), "c")
             

        h_count = s_list.to_string().count("h")
        b_count = s_list.to_string().count("b")
        c_count = s_list.to_string().count("c")
        prediction_struct = gorp.SecStructPrediction(pdb_code=test_p,
                                         protein=a_list.to_string(header=False, index=False).replace("\n", ""),
                                         prediction=tot_pred,
                                         real_sec_structure=s_list.to_string(header=False, index=False).replace("\n", ""),
                                         q3 = acc / len(a_list),
                                         q3_h = (tp_h / h_count) if h_count > 0 else 1,
                                         q3_b = (tp_b / b_count) if b_count > 0 else 1,
                                         q3_c = (tp_c / c_count) if c_count > 0 else 1,
                                         mcc_h=mcc_h,
                                         mcc_b=mcc_b,
                                         mcc_c=mcc_c,
                                         mcc=(mcc_h + mcc_b + mcc_c)/3
                                         )

        prediction_list.append(prediction_struct)
        tot_count += len(a_list)
        tot_acc += acc
        # Print the current prediction and the cumulated metrics.
        print(prediction_struct)    
        print("(", test_p_i + 1,"/", len(list(test_protein_set)), ") -- TOTAL ACCURACY:", "{:.3f}".format(tot_acc / tot_count))

#%%
        
    # Save predictions in a file.
    result_frame_gor_5_validation = pd.DataFrame(index=range(len(prediction_list)), columns=["PDB_code_and_chain",
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
                                                                                "q3_h",
                                                                                "q3_b",
                                                                                "q3_c",
                                                                                "mcc_h",
                                                                                "mcc_b",
                                                                                "mcc_c",
                                                                                "mcc",
                                                                                "overall_pred",
                                                                                "overall_structure_real"])
    for p_i, p in enumerate(prediction_list):
        result_frame_gor_5_validation.iloc[p_i, 0] = p.pdb_code
        result_frame_gor_5_validation.iloc[p_i, 1] = len(p.prediction)
        result_frame_gor_5_validation.iloc[p_i, 2] = p.prediction
        result_frame_gor_5_validation.iloc[p_i, 3] = p.real_sec_structure
        result_frame_gor_5_validation.iloc[p_i, 4] = p.prediction.count("h")
        result_frame_gor_5_validation.iloc[p_i, 5] = p.prediction.count("b")
        result_frame_gor_5_validation.iloc[p_i, 6] = p.prediction.count("c")
        result_frame_gor_5_validation.iloc[p_i, 7] = p.real_sec_structure.count("h")
        result_frame_gor_5_validation.iloc[p_i, 8] = p.real_sec_structure.count("b")
        result_frame_gor_5_validation.iloc[p_i, 9] = p.real_sec_structure.count("c")
        result_frame_gor_5_validation.iloc[p_i, 10] = p.protein.count("a")
        result_frame_gor_5_validation.iloc[p_i, 11] = p.protein.count("r")
        result_frame_gor_5_validation.iloc[p_i, 12] = p.protein.count("n")
        result_frame_gor_5_validation.iloc[p_i, 13] = p.protein.count("d")
        result_frame_gor_5_validation.iloc[p_i, 14] = p.protein.count("c")
        result_frame_gor_5_validation.iloc[p_i, 15] = p.protein.count("q")
        result_frame_gor_5_validation.iloc[p_i, 16] = p.protein.count("e")
        result_frame_gor_5_validation.iloc[p_i, 17] = p.protein.count("g")
        result_frame_gor_5_validation.iloc[p_i, 18] = p.protein.count("h")
        result_frame_gor_5_validation.iloc[p_i, 19] = p.protein.count("i")
        result_frame_gor_5_validation.iloc[p_i, 20] = p.protein.count("l")
        result_frame_gor_5_validation.iloc[p_i, 21] = p.protein.count("k")
        result_frame_gor_5_validation.iloc[p_i, 22] = p.protein.count("m")
        result_frame_gor_5_validation.iloc[p_i, 23] = p.protein.count("f")
        result_frame_gor_5_validation.iloc[p_i, 24] = p.protein.count("p")
        result_frame_gor_5_validation.iloc[p_i, 25] = p.protein.count("s")
        result_frame_gor_5_validation.iloc[p_i, 26] = p.protein.count("t")
        result_frame_gor_5_validation.iloc[p_i, 27] = p.protein.count("w")
        result_frame_gor_5_validation.iloc[p_i, 28] = p.protein.count("y")
        result_frame_gor_5_validation.iloc[p_i, 29] = p.protein.count("v")
        result_frame_gor_5_validation.iloc[p_i, 30] = p.protein
        result_frame_gor_5_validation.iloc[p_i, 31] = p.q3
        result_frame_gor_5_validation.iloc[p_i, 32] = p.q3_h
        result_frame_gor_5_validation.iloc[p_i, 33] = p.q3_b
        result_frame_gor_5_validation.iloc[p_i, 34] = p.q3_c
        result_frame_gor_5_validation.iloc[p_i, 35] = p.mcc_h
        result_frame_gor_5_validation.iloc[p_i, 36] = p.mcc_b
        result_frame_gor_5_validation.iloc[p_i, 37] = p.mcc_c
        result_frame_gor_5_validation.iloc[p_i, 38] = p.mcc
        result_frame_gor_5_validation.iloc[p_i, 39] = p.overall_prediction
        result_frame_gor_5_validation.iloc[p_i, 40] = test_proteins_input.loc[test_proteins_input["PDB_code_and_chain"] == p.pdb_code].overall_structure.iloc[0]
    result_frame_gor_5_validation.to_csv("../data/test_proteins_pred_result_gor_5_" + data_type + ".csv", index=False)



