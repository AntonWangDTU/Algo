#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# Weight Matrix construction including pseudo counts and sequence weighting
# 
# Some parts of the code have been blanked out. Fill out these places to make the code run. 

# ## Python Imports

# In[125]:


import numpy as np
import math
import copy
from pprint import pprint

#get_ipython().run_line_magic('matplotlib', 'inline')
from argparse import ArgumentParser

# We will now create an argument parser that will receive as arguments two values
# 1) the peptides-targets file to open (-f option)
# 2) the threshold to be applied in the target value filtering step (-t option)
# To achieve this, add the following lines below the ArgumentParser import line:


parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-b", action="store", dest="beta", type=int, default=0, help="Beta value for pseudo count (default: 0)")
parser.add_argument("-f", action="store", dest="peptides_targets_file", type=str, help="Peptides-Targets file")
parser.add_argument("-w", action="store", dest="weight", type=bool, default = False, help="State if you want to use seq weigth or not")
args = parser.parse_args()
beta = args.beta
sequence_weighting = args.weight
peptides_file = args.peptides_targets_file

# ## DEFINE THE PATH TO YOUR COURSE DATA DIRECTORY

# In[126]:


data_dir = "../../data/"


# ## Define options for run

# In[127]:


sequence_weighting = False
#sequence_weighting = False
# define weight on pseudo count
#beta = 0


# ## Data Imports

# ### Load Alphabet

# In[128]:


alphabet_file = data_dir + "Matrices/alphabet"

alphabet = np.loadtxt(alphabet_file, dtype=str)

print (alphabet)
print (len(alphabet))


# ### Load Background Frequencies

# In[129]:


bg_file = data_dir + "Matrices/bg.freq.fmt"
_bg = np.loadtxt(bg_file, dtype=float)

bg = {}
for i in range(0, len(alphabet)):
    bg[alphabet[i]] = _bg[i]

bg


# ### Load Blosum62 Matrix
# 

# In[130]:


blosum62_file = data_dir + "Matrices/blosum62.freq_rownorm"
_blosum62 = np.loadtxt(blosum62_file, dtype=float).T

blosum62 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum62[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum62[letter_1][letter_2] = _blosum62[i, j]

blosum62


# ### Load Peptides

# In[148]:


#peptides_file = data_dir + "PSSM/A0201.single_lig"
#peptides_file = data_dir + "PSSM/A0201.small_lig"
#peptides_file = data_dir + "PSSM/A0201.large_lig"

peptides = np.loadtxt(peptides_file, dtype=str).tolist()

if len(peptides[0]) == 1:
    peptide_length = len(peptides)
    peptides = [peptides]
else:
    peptide_length = len(peptides[0])

for i in range(0, len(peptides)):
    if len(peptides[i]) != peptide_length:
        print("Error, peptides differ in length!")
        
print(peptides)
peptide_length


# ## Initialize Matrix

# In[132]:


def initialize_matrix(peptide_length, alphabet):

    init_matrix = [0]*peptide_length

    for i in range(0, peptide_length):

        row = {}

        for letter in alphabet: 
            row[letter] = 0.0

        #fancy way:  row = dict( zip( alphabet, [0.0]*len(alphabet) ) )

        init_matrix[i] = row
        
    return init_matrix


# ## Amino Acid Count Matrix (c)

# In[133]:


c_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
        
    for peptide in peptides:
        
        c_matrix[position][peptide[position]] += 1
    
#pprint(c_matrix)
c_matrix[0]["A"]


# ## Sequence Weighting

# In[134]:


# w = 1 / r * s
# where 
# r = number of different amino acids in column
# s = number of occurrence of amino acid in column

weights = {}

for peptide in peptides:

    # apply sequence weighting
    if sequence_weighting:
    
        w = 0.0
        neff = 0.0
        
        for position in range(0, peptide_length):

            r = 0

            for letter in alphabet:        

                if c_matrix[position][letter] != 0:
                    
                    r += 1

            s = c_matrix[position][peptide[position]]

            w += 1.0/(r * s)

            neff += r
                
        neff = neff / peptide_length
  
    # do not apply sequence weighting
    else:
        
        w = 1  
        
        neff = len(peptides)  
      

    weights[peptide] = w

pprint( "W:")
pprint( weights )
pprint( "Nseq:")
pprint( neff )


# ## Observed Frequencies Matrix (f)

# In[135]:


f_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
  
    n = 0;
  
    for peptide in peptides:
    
        f_matrix[position][peptide[position]] += weights[peptide]
    
        n += weights[peptide]
        
    for letter in alphabet: 
        
        f_matrix[position][letter] = f_matrix[position][letter]/n
      
pprint( f_matrix[0] )


# In[ ]:





# ## Pseudo Frequencies Matrix (g)
# 
# Remember g(b) = sum f(a)* q(b|a), and blosum[a,b] = q(a|b)

# In[136]:


g_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):

    for letter_1 in alphabet:
        for letter_2 in alphabet:
        
          g_matrix[position][letter_1] += f_matrix[position][letter_2] * blosum62[letter_1][letter_2]

pprint(g_matrix[0])


# ## Combined Frequencies Matrix (p)

# ![image.png](attachment:image.png)

# In[137]:


p_matrix = initialize_matrix(peptide_length, alphabet)

alpha = neff - 1

for position in range(0, peptide_length):

    for a in alphabet:
        p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a])/(alpha + beta)

pprint(p_matrix[0])


# ## Log Odds Weight Matrix (w)

# In[138]:


w_matrix = initialize_matrix(peptide_length, alphabet)

for position in range(0, peptide_length):
    
    for letter in alphabet:
        if p_matrix[position][letter] > 0:
            w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter]/bg[letter])/math.log(2)
        else:
            w_matrix[position][letter] = -999.9

pprint(w_matrix[0])


# ### Write Matrix to PSI-BLAST format

# In[139]:


def to_psi_blast(matrix):

    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    print ('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for i, row in enumerate(matrix):

        scores = []

        scores.append(str(i+1) + " A")

        for letter in letter_order:

            score = row[letter]

            scores.append(round(score, 4))

        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 


# ### convert w_matrix to PSI-BLAST format and print to file

# In[140]:


def to_psi_blast_file(matrix, file_name):
    
    with open(file_name, 'w') as file:

        header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        file.write ('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n'.format(*header)) 

        letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        for i, row in enumerate(matrix):

            scores = []

            scores.append(str(i+1) + " A")

            for letter in letter_order:

                score = row[letter]

                scores.append(round(score, 4))

            file.write('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n'.format(*scores)) 


# ### convert  w_matrix to PSI-BLAST format

# In[141]:


to_psi_blast(w_matrix)


# ### convert w_matrix to PSI-BLAST format and print to file

# In[142]:


# Write out PSSM in Psi-Blast format to file
file_name = "w_matrix_test"
to_psi_blast_file(w_matrix, file_name)


# ## Evaluation

# In[143]:


#evaluation_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_2/A0201.eval"
evaluation_file = data_dir + "PSSM/A0201.eval"
#evaluation_file = evaluation_upload.values()

evaluation = np.loadtxt(evaluation_file, dtype=str).reshape(-1,2)
evaluation_peptides = evaluation[:, 0]
evaluation_targets = evaluation[:, 1].astype(float)

evaluation_peptides, evaluation_targets


# In[144]:


def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum


# In[145]:


evaluation_predictions = []
for evaluation_peptide in evaluation_peptides:
    evaluation_predictions.append(score_peptide(evaluation_peptide, w_matrix))


# In[146]:


from scipy.stats import pearsonr
import matplotlib.pyplot as plt

pcc = pearsonr(evaluation_targets, evaluation_predictions)
print("PCC: ", pcc[0])

plt.scatter(evaluation_targets, evaluation_predictions);


# In[ ]:




