import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import glob
import pprint
import warnings
import sys
import re

from matplotlib.patches import Rectangle
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted

# use readline to read line by line
if len(sys.argv) < 2:
    print('Usage: python read_data_logfile.py <logfile>')
    exit(1)

filename = sys.argv[1]
file = open(filename, 'r')
line_content = file.readlines()

# arrays for tracking needed information
arr_trial_indices = []
arr_exp_vars = [0.0] * 100
 
tmp_idx = 0
line_count = 0
# Strips the newline character
for line in line_content:
    line_count += 1
    
    if 'Params for Trial' in line:
        line_tokens = line.split(' ')
        print("Line {}: {}".format(line_count, line_tokens))
        
        trial_idx = int(re.findall(r"\d+", line_tokens[-1])[0]) if len(line_tokens) == 4 else \
                    int(re.findall(r"\d+", line_tokens[3])[0])
        if trial_idx < 100:
            arr_trial_indices.append(trial_idx)

        tmp_idx = trial_idx
    
    if 'Average early_stopping_point' in line:
        line_tokens = line.split(' ')
        print("Line {}: {}".format(line_count, line_tokens))
        
        expvar = float(re.findall(r"[-+]?(?:\d*\.*\d+)", line_tokens[3])[0])
        if tmp_idx < 100:
            arr_exp_vars[tmp_idx] = expvar

print('Trials: lenght={}, {}'.format(len(arr_trial_indices), arr_trial_indices))
print('---------------------------------------------------------------------\n')
print('Expvar: lenght={}, {}'.format(len(arr_exp_vars), arr_exp_vars))
