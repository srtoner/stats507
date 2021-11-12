import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.stats as stats
import timeit
import pickle
from IPython.core.display import display, HTML
demo_cohorts = []
oh_cohorts = []
prefix = "https://wwwn.cdc.gov/Nchs/Nhanes/"
years = "2011-2012"
demo_file = "/DEMO_"
oh_file = "/OHXDEN_"
ext = ".XPT"
letters = ['G', 'H', 'I', 'J']

demo_columns = ["Cohort", 
                    "SEQN", 
                    "RIDAGEYR",
                    "RIAGENDR",
                    "RIDRETH3", 
                    "DMDEDUC2", 
                    "DMDMARTL",
                    "RIDSTATR", 
                    "SDMVPSU", 
                    "SDMVSTRA", 
                    "WTMEC2YR", 
                    "WTINT2YR"]

demo_labels = { 
                "SEQN" : "id", 
                "RIDAGEYR" : "age",
                "RIAGENDR" : "gender",
                "RIDRETH3" : "ethnicity", 
                "DMDEDUC2" : "education", 
                "DMDMARTL" : "marital_status",
                "RIDSTATR" : "exam_status", 
                "SDMVPSU" : "masked_var_unit", 
                "SDMVSTRA" : "pseudo_strat", 
                "WTMEC2YR" : "interview_weight", 
                "WTINT2YR" : "exam_weight"
                }
    

oh_columns = ["Cohort", "SEQN", "OHDDESTS"]
oh_labels = {"SEQN" : "id", "OHDDESTS" : "dental_status"}

oh_skip_list = [1,16,17,32] # these numbers should be skipped for condition var

for i in range(1,33):
    oh_prefix = "OHX"
    count_suffix = "TC"
    cond_suffix = "CTC"
    i_str = str(i)
    if i < 10:
        i_str = "0"+i_str
    
    oh_columns.append(oh_prefix + i_str + count_suffix)
    oh_labels[oh_prefix + i_str + count_suffix] = "tooth_count" + i_str
   
    if(i not in oh_skip_list):
        oh_labels[oh_prefix + i_str + cond_suffix] = "tooth_cond" + i_str
        oh_columns.append(oh_prefix + i_str + cond_suffix)

j = 0
for i in range(2011, 2019, 2): 
    year = str(i) + "-" + str(i + 1)
    
    demo_url = prefix + year + demo_file + letters[j] + ext
    oh_url = prefix + year + oh_file + letters[j] + ext
    j += 1
    demo_new_df = pd.read_sas(demo_url)
    oh_new_df = pd.read_sas(oh_url)
    
    demo_new_df["Cohort"] = year
    oh_new_df["Cohort"] = year
    demo_cohorts.append(demo_new_df)
    oh_cohorts.append(oh_new_df)   
# - 

# Now that we have retrieved the data, it's time to consolidate it into two
# separate data frames, fix labels, and then cast the variables to specific 
# data types. For the most part, we will be casting to `str` as most of the 
# dental data set has categorical data, indicated by integer values.

# + 
demo_data = pd.concat(demo_cohorts)
demo_data = demo_data[demo_columns]
demo_data = demo_data.rename(columns=demo_labels)

oh_data = pd.concat(oh_cohorts)
oh_data = oh_data[oh_columns]
oh_data = oh_data.rename(columns=oh_labels)

# Cast various columns to appropriate datatype
demo_val_list = list(demo_labels.values())
demo_types = dict()
for i in demo_val_list:
    demo_types[i] = str

demo_types["interview_weight"] = float
demo_types["exam_weight"] = float
demo_types["age"] = int

demo_data = demo_data.astype(demo_types)
# -

# Lastly, we can get the number of observations by counting the number of rows:
    
print("The demographic dataset has " + str(len(demo_data.index)) + " cases.")
print("The dental dataset has " + str(len(oh_data.index)) + " cases.")

# Now we just need to save the data to a round-trip format, in this case 
# using the parquet format.

demo_file = "nhanes_demo.pkl"
dental_file = "nhanes_dental.pkl"

dem_file = open(demo_file, "wb")
pickle.dump(demo_data, dem_file)
dem_file.close()

dent_file = open(dental_file, "wb")
pickle.dump(oh_data, dent_file)
dent_file.close()
