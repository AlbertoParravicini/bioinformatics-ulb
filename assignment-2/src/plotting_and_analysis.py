import plotly
import numpy as np
import pandas as pd

# File name
file_name = "../data/pred_result_dssp_2.csv"
# Read the prediction data
pred_data_dssp = pd.read_csv(file_name)

# File name
file_name = "../data/pred_result_stride_2.csv"
# Read the prediction data
pred_data_stride = pd.read_csv(file_name)

pred_data_dssp["type"] = "dssp"
pred_data_stride["type"] = "stride"
# Concatenate the data, useful for plotting
pred_data_tot = pd.concat([pred_data_dssp, pred_data_stride], ignore_index=True)

#%%

c = sum((a=="b" and b=="b") for a, b in zip(pred_data_dssp.at[0, "prediction"], pred_data_dssp.at[0, "real_sec_structure"]))
perc = c / pred_data_dssp.at[0, "real_sec_structure"].count("b")