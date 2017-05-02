import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D
from keras.layers import Embedding
from keras.layers import GRU
from keras.preprocessing import text
from keras.layers.core import Activation  
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed


from keras.utils.np_utils import to_categorical

aminoacid_list = "arndceqghilkmfpstwyv-"
sec_struct_list = "hbc-"

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

# Map aminoacid to numbers and viceversa.
a_to_i = {a: i for i, a in enumerate(list(aminoacid_list))}
i_to_a = {i: a for i, a in enumerate(list(aminoacid_list))}
# Map secondary structures to numbers and viceversa.
s_to_i = {s: i for i, s in enumerate(list(sec_struct_list))}
i_to_s = {i: s for i, s in enumerate(list(sec_struct_list))}

#%%

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

def build_dataset(input_data, max_len=20):
    # List of proteins
    protein_set = set(input_data.PDB_code_and_chain)
    
    # Build the matrices used as dataset.
    X = np.zeros(shape=(len(protein_set), max_len))
    Y = np.zeros(shape=(len(protein_set), max_len))
    
    for p_i, p in enumerate(protein_set):
        # Extract the aminoacid and secondary structures for each protein.
        a_seq = input_data.loc[input_data["PDB_code_and_chain"] == p].a
        s_seq = input_data.loc[input_data["PDB_code_and_chain"] == p].s
        # Turn the series to strings.
        a_seq = a_seq.to_string(header=False, index=False).replace("\n", "")
        s_seq = s_seq.to_string(header=False, index=False).replace("\n", "")
        
        a_seq = a_seq[0:max_len] if len(a_seq) > max_len else (a_seq + "-" * (max_len - len(a_seq)))
        s_seq = s_seq[0:max_len] if len(s_seq) > max_len else (s_seq + "-" * (max_len - len(s_seq)))
        
        X[p_i, :] = [a_to_i[x] for x in a_seq]
        Y[p_i, :] = [s_to_i[x] for x in s_seq]
        
    return [X, Y]    
        
    

#%% 
if __name__ == '__main__':
    
    # Type of the data to read ("stride", "dssp")
    data_type = "dssp"
    # File name
    file_name = "../data/" + data_type + "_info.txt"
    # Read the data
    input_data = preprocess_input(file_name, aminoacid_codes)
    
    # Max length of the aminoacid sequences used in the dataset.
    # Longer sequences are truncated.
    max_len = 40
    
    protein_num = len(set(input_data.PDB_code_and_chain))
    
    dataX, dataY = build_dataset(input_data, max_len)
    
    # reshape X to be [samples, time_steps, features]
    # In this case, we are using a 1 dimensional signal,
    # but we pass a vector, which has a time component. 
    X = np.reshape(dataX, (dataX.shape[0], max_len, 1))
    # Normalize X over [0, 1]
    X = X / float(len(aminoacid_list))
    # one hot encoding on the output variable. It will become a tensor (n X max_len X 4)
    Y = np.reshape(to_categorical(dataY), (len(dataY), max_len, 4))
    
    #%% Build train, test datasets
    
    # Build train and test sets
    print("building train and test sets")
    split_factor = 0.8
    train_size = int(X.shape[0] * split_factor)
    train_indices = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
    
    X_train = X[train_indices, :, :]
    Y_train = Y[train_indices, :, :]
    
    test_indices = np.array(list(set(range(X.shape[0])) - set(train_indices)))
    X_test = X[test_indices, :]
    Y_test = Y[test_indices, :, :]
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    
    #%% Create the model
    
    model = Sequential()
    model.add(GRU(40, activation="tanh", recurrent_dropout=0.1, dropout=0.1, input_shape=(None, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(len(sec_struct_list))))
    model.add(Activation("softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(model.summary())
    
    #%% Train 
    
    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit(X_train, Y_train, epochs=200, batch_size=40, callbacks=callbacks_list)
    # Final evaluation of the model
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("LOSS: %.2f%%" % (score))
    
    filepath = "model_1_lstm.h5"
    model.save(filepath)
    
    #%% predict
    filepath = "weights.best.hdf5"
    model.load_weights(filepath)
    
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    
#    a_seq_o = "HPKPSACRNLFGPVDHEELTRDLEKHCRDMEEASQRKWNFDFQNHKPLEGKYEWQEVEKGSLPEFYYRPPRPPKGACKVPAQES".lower()[0:max_len]
#    a_seq = np.array([a_to_i[x] for x in list(a_seq_o)])
#    a_seq = a_seq / float(len(aminoacid_list))
#       
#    a_seq = a_seq.reshape(1, max_len, 1)
#    
#    s_seq_o = "CCCCCCCCCCCCCCCHHHHHHHHHHHHCCCCHHHHHHHCEECCCCEECCCCCCCEEEECCCCCHHHHCCCCCCCCCCCCCCCCC".lower()[0:max_len]
#    s_seq = [s_to_i[x] for x in list(s_seq_o)]
#    s_seq = np.reshape(to_categorical(s_seq), (1, max_len, 3))
#    
#    out = model.predict(a_seq)   
#    pred = ""
#    for i in range(out.shape[1]):
#        pred += i_to_s[np.argmax(out[0, i, :])]

    out = model.predict(X_test)
    acc = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
           if np.argmax(out[i, j, :]) == np.argmax(Y_test[i, j, :]):
               acc +=1
    print("ACC:", acc / (out.shape[0] * out.shape[1]))
    
    # ACC: 0.43675
    # LOSS: 1.01070
