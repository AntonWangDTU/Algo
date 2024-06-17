#!/bin/env python3
# # Viterbi
# Implementation of the Viterbi algorithm.
# 
# 

# ## Python Imports

# In[67]:


import numpy as np


# ## Simulation Main

# ### Parameters

from argparse import ArgumentParser

# We will now create an argument parser that will receive as arguments two values
# 1) the peptides-targets file to open (-f option)
# 2) the threshold to be applied in the target value filtering step (-t option)
# To achieve this, add the following lines below the ArgumentParser import line:


parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-i", action="store", dest="INPUT_FILE", type=str, help="Input File")
args = parser.parse_args()
input_file = args.INPUT_FILE

input_sequence = np.loadtxt(input_file, dtype=object)

print(input_sequence)