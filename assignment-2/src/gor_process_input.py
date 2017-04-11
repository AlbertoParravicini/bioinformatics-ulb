#%% IMPORT DATA

import pandas as pd
import numpy as np
import timeit
import pickle

aminoacid_list = "arndceqghilkmfpstwyv"

# Dictionary that maps aminoacid codes to single letters.
aminoacid_codes = {"ala":  "a",
                   "arg":  "r",
                   "asn":  "n",
                   "asp":  "d",
                   "cys":  "c",
                   "gln":  "q",
                   "glu":  "e",
                   "gly":  "g",
                   "his":  "h",
                   "ile":  "i",
                   "leu":  "l",
                   "lys":  "k",
                   "met":  "m",
                   "phe":  "f",
                   "pro":  "p",
                   "ser":  "s",
                   "thr":  "t",
                   "trp":  "w",
                   "tyr":  "y",
                   "val":  "v"}


def preprocess_input(file_name, aminoacid_codes):
    """
    Load and preprocess a file containing protein sequences.
    The output will be a DataFrame ready to be used by GOR.
    :param file_name: string, name of the file to be opened.
    :param aminoacid_codes: dictionary that maps aminoacid codes to single letters.
    :return: DataFrame
    """
    # Load the data
    input_data = pd.read_csv(file_name,
                         header=None, sep="\t",
                         names=["PDB_code", "PDB_chain_code", "PDB_seq_code", "residue_name", "secondary_structure"])

    # CLEAN DATA - AMINOACIDS

    # There are some values that are weird, like "a", "b", ...
    # Some of those can be interpreted as correct aminoacids.

    # Remove rows with "X", "b", "UNK"
    # First, remove any leading spaces, then remove "X", "b", "UNK"
    input_data["residue_name"] = input_data["residue_name"].str.strip()
    input_data = input_data.drop(input_data[input_data["residue_name"].isin(["X", "b", "UNK"])].index).reset_index(drop=True)


    # Add a column for the aminoacids, while preserving the original one
    input_data["a"] = input_data["residue_name"]
    input_data["a"] = input_data["a"].str.lower()


    # replace the codes with something shorter
    input_data["a"] = input_data["a"].map(lambda x: aminoacid_codes[x] if x in aminoacid_codes else x)

    # Create a new column
    input_data["s"] = input_data["secondary_structure"]
    input_data["s"] = input_data["s"].str.lower()
    # Replace "other" with "coil"
    input_data.loc[input_data["s"] == "other", "s"] = "coil"
    # Shorten values
    input_data["s"] = input_data["s"].str[0]

    # Add a composite key to the dataset.
    # Append the PDB chain code to the PDB code, to obtain unique protein identifier.
    input_data["PDB_code_and_chain"] = input_data.PDB_code + "_" + input_data.PDB_chain_code

    return input_data


#%% Build S_j, A_j matrix

def build_sj_aj_matrix(s_list, a_list):
    """
    Count the number of times that each aminoacid a_j has secondary structure s_j.
    :param s_list: a list of secondary structures; pandas Series.
    :param a_list: a list of aminoacids; pandas Series.
    :return: pandas.DataFrame
    """
    # Set of all secondary structures.
    sec_structure_set = set(s_list)
    # Set of all aminoacids.
    aminoacid_set = set(a_list)
    # Occurrency matrix.
    sj_aj_matrix = pd.DataFrame(0, index=sec_structure_set, columns=aminoacid_set, dtype=int)
    # Zip together the lists of secondary structures and aminoacids, it makes counting easier.
    s_a_list = [list(x) for x in zip(s_list,a_list)]
    # Count the occurrencies.
    for s in sec_structure_set:
        for a in aminoacid_set:
            sj_aj_matrix.at[s, a] = s_a_list.count([s, a])

    return sj_aj_matrix



#%% Build S_j, A_j, A_jm matrix

def build_sj_aj_ajm(input_data, print_details=True):
    """
    Count the times where aminoacid a_j appears with secondary structure s_j,
    and aminoacid a_jm appears at a distance m from them, with m in [-8, 8].
    """
    # Set of all proteins.
    protein_set = list(set(input_data.PDB_code_and_chain))
    # Set of all secondary structures.
    sec_structure_set = list(set(input_data.s))
    # Set of all aminoacids.
    aminoacid_set = list(set(input_data.a))
    # Build a dictionary where the keys are the secondary structures, and the values are 3D tensors
    # with index a_j, a_jm, m.
    sj_aj_ajm_dict = {s: pd.Panel(data=0, items=aminoacid_set, major_axis=aminoacid_set, minor_axis=np.arange(-8, 9), dtype=int) for s in sec_structure_set}

    # The counting must be done separately for each protein,
    # otherwise the aminoacids at the end of one protein would be counted
    # as part of the next one!
    for i_p, p in enumerate(protein_set):
        s_list = input_data.loc[input_data['PDB_code_and_chain'] == p].s
        a_list = input_data.loc[input_data['PDB_code_and_chain'] == p].a
        n = len(s_list)
        for i_aj, a_j in enumerate(aminoacid_set):
            for i_ajm, a_jm in enumerate(aminoacid_set):
                for i_m, m in enumerate(np.arange(-8, 9)):
                    if print_details:
                        print(p, a_j, a_jm, m, "-------", i_p / len(protein_set), i_aj / 20, i_ajm / 20)

                    for i in range(max(0, -m), min(n, n - m)):
                        if a_list.iat[i] == a_j and a_list.iat[i + m] == a_jm:
                            # Increment the count of the right dictionary.

                            sj_aj_ajm_dict[s_list.iat[i]].iat[i_aj, i_ajm, i_m] += 1

    return sj_aj_ajm_dict



###################
# OUTDATED ########
###################

def count_residues_gor_3(s_list, a_list, s_j, a_j, a_jm, lag):
    """
    Count the number of times that structure "s_j" appears together with residue "a_j",
    and residue "a_jm" is at distance "lag" from them.
    :param s_list: a list of secondary structures; pandas Series.
    :param a_list: a list of aminoacids; pandas Series.
    :param s_j: a secondary structure name
    :param a_j: an aminoacid name
    :param a_jm: an aminoacid name
    :param lag: distance from a_j at which a_jm is searched
    :return: int
    """
    n = len(s_list)
    count = 0
    for i in range(max(0, -lag), min(n, n - lag)):
        if s_list.iat[i] == s_j and a_list.iat[i] == a_j and a_list.iat[i + lag] == a_jm:
            count += 1
    return count


def count_residues_gor_1(s_list, a_list, s_j, a_jm, lag):
    """
    Count the number of times that structure "s_j" appears,
    and residue "a_jm" is at distance "lag" from them.
    :param s_list: a list of secondary structures; pandas Series.
    :param a_list: a list of aminoacids; pandas Series.
    :param s_j: a secondary structure name
    :param a_jm: an aminoacid name
    :param lag: distance from a_j at which a_jm is searched
    :return: int
    """
    n = len(s_list)
    count = 0
    for i in range(max(0, -lag), min(n, n - lag)):
        if s_list.iloc[i] == s_j and a_list.iloc[i + lag] == a_jm:
            count += 1
    return count

def count_aminoacid_lag(a_list, a_j, a_jm, lag):
    """
    Count the number of times aminoacid "a_jm" appears at distance "lag" from aminoacid "a_j".
    :param a_list: a list of aminoacids; pandas Series.
    :param a_j: an aminoacid name
    :param a_jm: an aminoacid name
    :param lag: distance from a_j at which a_jm is searched
    :return: int
    """
    n = len(a_list)
    count = 0
    for i in range(max(0, -lag), min(n, n - lag)):
        if a_list.iloc[i] == a_j and a_list.iloc[i + lag] == a_jm:
            count += 1
    return count

def info_value_gor_1(s_list, a_list, s_j, a_jm, lag, a_occ):
    """
    Compute the information value associated to a secondary structure s_j, in position j,
    and to an aminoacid a_jm, at distance m from s_j.
    :param s_list: a list of secondary structures; pandas Series.
    :param a_list: a list of aminoacids; pandas Series.
    :param s_j: a secondary structure name
    :param a_jm: an aminoacid name
    :param lag: distance from a_j at which a_jm is searched
    :param a_occ: number of occurrencies of each aminoacid
    :return: double
    """
    num_1 = count_residues_gor_1(s_list, a_list, s_j, a_jm, lag)
    den_1 = a_occ[a_jm] - num_1
    den_2 = s_occ[s_j]
    num_2 = len(s_list) - den_2
    return np.log(num_1 * num_2) - np.log(den_1 * den_2)          

def info_value_gor_3(s_list, a_list, s_j, a_j, a_jm, lag, a_occ):
    """
    Compute the information value associated to an aminoacid a_j in position j,
    with secondary structure s_j, in position j,
    and to an aminoacid a_jm, at distance m from s_j.
    :param s_list: a list of secondary structures; pandas Series.
    :param a_list: a list of aminoacids; pandas Series.
    :param s_j: a secondary structure name
    :param a_j: an aminoacid name
    :param a_jm: an aminoacid name
    :param lag: distance from a_j at which a_jm is searched
    :param a_occ: number of occurrencies of each aminoacid
    :return: double
    """
    # Num of times s_j appears with a_j, and a_jm is lag position distant.
    num_1 = count_residues_gor_3(s_list, a_list, s_j, a_j, a_jm, lag)
    # Num of times a_j appears with a secondary structure different from s_j,
    # and a_jm is lag position distant.
    den_1 = count_aminoacid_lag(a_list, a_j, a_jm, lag) - num_1
    
    # Number of times aminoacid a_j appears together with structure s_j.                        
    den_2 = count_residues_gor_3(s_list, a_list, s_j, a_j, a_j, 0)                          
    # Number of times aminoacid a_j appears with a different structure from s_j.
    num_2 = a_occ[a_j] - den_2
                 
    info = np.log(num_1) - np.log(den_1) + np.log(num_2) - np.log(den_2)   
    return info     

###################
# END OF OUTDATED #
###################

#%% GOR

def compute_gor_3(a_list, i, sj_aj_ajm_dict, sj_aj_matrix, a_occ, l_1_out=False):
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

            # Num of times s_j appears with a_j, and a_jm is lag position distant.
            num_1 = sj_aj_ajm_dict[s].at[a_j, a_jm, m] - (1 if l_1_out else 0)
            # Num of times a_j appears with a secondary structure different from s_j,
            # and a_jm is lag position distant.
            den_1 = sum([sj_aj_ajm_dict[s_i].at[a_j, a_jm, m] for s_i in info_tot.keys()]) - num_1 + (1 if l_1_out else 0)

            # Number of times aminoacid a_j appears together with structure s_j.
            den_2 = sj_aj_matrix.at[s, a_j] - (1 if l_1_out else 0)
            # Number of times aminoacid a_j appears with a different structure from s_j.
            num_2 = a_occ.at[a_j] - den_2 + (1 if l_1_out else 0)

            # Add up the information value.
            info_tot[s] += np.log(num_1) - np.log(den_1) + np.log(num_2) - np.log(den_2)

    # Predict the secondary structure with highest value.
    return [max(info_tot, key=info_tot.get), max(info_tot.values())]





####################################
# TESTS ############################
####################################

if __name__ == '__main__':
    # Type of the data to read ("stride", "dssp")
    data_type = "dssp"
    # File name
    file_name = "../data/" + data_type + "_info.txt"
    # Read the data
    input_data = preprocess_input(file_name, aminoacid_codes)

    # Look at the aminoacid values
    aminoacids = set(input_data.residue_name)
    print(aminoacids)
    # Same stuff with the secondary structures
    secondary_structures = set(input_data.secondary_structure)
    print(secondary_structures)

    # Number of residues
    n_res = len(input_data.index)

    # Compute the number of occurrencies of each aminoacid
    a_occ = input_data["a"].value_counts()

    # Compute the number of occurrencies of each secondary structure
    s_occ = input_data["s"].value_counts()

    # Count how many times each aminoacid has a given secondary structure.
    start_time = timeit.default_timer()
    sj_aj_matrix = build_sj_aj_matrix(input_data.s, input_data.a)
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF build_sj_aj_matrix:", (end_time - start_time), "\n")


    # Count the times where aminoacid a_j appears with secondary structure s_j,
    # and aminoacid a_jm appears at a distance m from them, with m in [-8, 8].
    #   WARNING: this can take a bit of time, usually about 20-30 minutes;
    #            do it at your own risk!
    start_time = timeit.default_timer()
    ##########################
    # UNCOMMENT IF NEEDED ####
    ##########################
    sj_aj_ajm_dict = build_sj_aj_ajm(input_data)
    end_time = timeit.default_timer()
    print("! -> EXECUTION TIME OF build_sj_aj_ajm:", (end_time - start_time), "\n")

    # Save data
    dict_file_name = "sj_aj_ajm_dict_"+ data_type + ".p"

    # Open the file for writing.
    file_object = open(dict_file_name.encode('utf-8').strip(), 'wb')
    # Save data
    pickle.dump(sj_aj_ajm_dict, file_object)

    file_object.close()

    # Load data
    file_object = open(dict_file_name.encode('utf-8').strip(), 'rb')
    sj_aj_ajm_dict = pickle.load(file_object)
    
    

