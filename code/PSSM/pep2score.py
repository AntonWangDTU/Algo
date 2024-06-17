#!/usr/bin/env python
# coding: utf-8

# # Description
# Scoring peptides to a weight matrix

# ## Python Imports

# In[1]:


import numpy as np
from pprint import pprint

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')
from argparse import ArgumentParser

# ## DEFINE THE PATH TO YOUR COURSE DATA DIRECTORY

parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-mat", action="store", dest="PSSM_file", type=str, help="File containing a PSSM")
parser.add_argument("-f", action="store", dest="peptides_targets_file", type=str, help="Peptides-Targets file")

args = parser.parse_args()
mat = args.PSSM_file
peptides_targets_file = args.peptides_targets_file


data_dir = "../../data/"


# ## Initialize Matrix

# In[3]:


def initialize_matrix(peptide_length, alphabet):

    init_matrix = [0]*peptide_length

    for i in range(0, peptide_length):

        row = {}

        for letter in alphabet: 
            row[letter] = 0.0

        #fancy way:  row = dict( zip( alphabet, [0.0]*len(alphabet) ) )

        init_matrix[i] = row
        
    return init_matrix


# ### Load Matrix from PSI-BLAST format

# In[4]:


def from_psi_blast(file_name):

    f = open(file_name, "r")
    
    nline = 0
    for line in f:
    
        sline = str.split( line )
        
        if nline == 0:
        # recover alphabet
            alphabet = [str]*len(sline)
            for i in range(0, len(sline)):
                alphabet[i] = sline[i]
                
            matrix = initialize_matrix(peptide_length, alphabet)
        
        else:
            i = int(sline[0])
            
            for j in range(2,len(sline)):
                matrix[i-1][alphabet[j-2]] = float(sline[j])
                
        nline+= 1
            
    return matrix


# ### Score peptide to mat

# In[5]:


def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum


# ## Main

# In[6]:


# Read evaluation data
#evaluation_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_2/A0201.eval"
evaluation_file = peptides_targets_file

evaluation = np.loadtxt(evaluation_file, dtype=str)
evaluation_peptides = evaluation[:, 0]
evaluation_targets = evaluation[:, 1].astype(float)

evaluation_peptides, evaluation_targets

peptide_length = len(evaluation_peptides[0])

# Define which PSSM file to use (file save from pep2mat)
pssm_file = mat

w_matrix = from_psi_blast(pssm_file)

evaluation_predictions = []
for i in range(len(evaluation_peptides)):
    score = score_peptide(evaluation_peptides[i], w_matrix)
    evaluation_predictions.append(score)
    print (evaluation_peptides[i], score, evaluation_targets[i])
    
pcc = pearsonr(evaluation_targets, evaluation_predictions)
print("PCC: ", pcc[0])

plt.scatter(evaluation_targets, evaluation_predictions);


# In[ ]:




