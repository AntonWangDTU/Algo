#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# Smith-Waterman algorithm in O2 time.
# 
# Parts of the code has been blanked out with X's. Fill in these parts to make the code run 

# ## Python Imports

# In[6]:


import numpy as np


# ## Data Imports

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[7]:


data_dir = "../../data/"


# ## Define runtime parameters



#get_ipython().run_line_magic('matplotlib', 'inline')
from argparse import ArgumentParser

# We will now create an argument parser that will receive as arguments two values
# 1) the peptides-targets file to open (-f option)
# 2) the threshold to be applied in the target value filtering step (-t option)
# To achieve this, add the following lines below the ArgumentParser import line:


parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-q", action="store", dest="query", type=str, default = "VLPVLILP", help="File with query sequence")
parser.add_argument("-db", action="store", dest="database", type=str, default = "VLLPVLLP", help="File with database sequence")
parser.add_argument("-go", action="store", dest="GAP_OPEN", type=int, default = -11.0, help="Value of gap open (-11.0)")
parser.add_argument("-ge", action="store", dest="GAP_EXTENSION", type=int, default = -1, help="Value of gap extension (-1.0)")
args = parser.parse_args()
query = args.query
database = args.database
gap_open = args.GAP_OPEN
gap_extension = args.GAP_EXTENSION
# In[8]:


#gap_open = -11
#gap_extension = -1


# ### Alphabet

# In[9]:


#alphabet_file = alphabet_upload.values()
#alphabet_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/alphabet"
alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)

alphabet


# ### Blosum Matrix

# In[11]:


#blosum_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/blosum50"
blosum_file = data_dir + "Matrices/BLOSUM50"
_blosum50 = np.loadtxt(blosum_file, dtype=float).reshape((24, -1)).T

blosum50 = {}

for i, letter_1 in enumerate(alphabet):
    
    blosum50[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        blosum50[letter_1][letter_2] = _blosum50[i, j]

#blosum50


# ## Alignment Matrix
# 
# This functions returns, apart from the final Alignment Matrix, all the intermedite Matrices (for plotting purposes).
# 
# 

# In[17]:


def smith_waterman_alignment(query="VLLP", database="VLILP", scoring_scheme={}, gap_open=-5, gap_extension=-1):

    # Matrix imensions
    M = len(query)
    N = len(database)
    
    # D matrix change to float
    D_matrix = np.zeros((M+1, N+1), dtype = int)

    # P matrix
    P_matrix = np.zeros((M+1, N+1), dtype = int)
    
    # Q matrix
    Q_matrix = np.zeros((M+1, N+1), dtype = int)

    # E matrix
    E_matrix = np.zeros((M+1, N+1), dtype=object)

    # Initialize matrices
    for i in range(M, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        # alignment_matrix[i-1, N] = alignment_matrix[i, N] + gap_open
        D_matrix[i-1, N] = 0
        P_matrix[i-1, N] = 0
        Q_matrix[i-1, N] = 0
        E_matrix[i-1, N] = 0

    for j in range(N, 0, -1):
        # Here you might include  penalties for end gaps, i.e
        #alignment_matrix[M, j-1] = alignment_matrix[M, j] + gap_open
        D_matrix[M, j-1] = 0
        P_matrix[M, j-1] = 0
        Q_matrix[M, j-1] = 0
        E_matrix[M, j-1] = 0

        
    # Main loop
    D_matrix_max_score, D_matrix_i_max, D_matrix_i_max = -9, -9, -9
    for i in range(M-1, -1, -1): 
        for j in range(N-1, -1, -1):
            
            # Q_matrix[i,j] entry
            gap_open_database = D_matrix[i+1, j] + gap_open
            gap_extension_database = Q_matrix[i+1, j] + gap_extension
            max_gap_database = max(gap_open_database, gap_extension_database)
            
            Q_matrix[i,j] = max_gap_database
                
            # P_matrix[i,j] entry
            gap_open_query = D_matrix[i,j+1] + gap_open
            gap_extension_query = P_matrix[i, j+1] + gap_extension
            max_gap_query = max(gap_open_query, gap_extension_query)
            
            P_matrix[i,j] = max_gap_query
            
            # D_matrix[i,j] entry
            diagonal_score = D_matrix[i+1,j+1] + blosum50[query[i]][database[j]]         
            
            # E_matrix[i,j] entry
            candidates = [(1, diagonal_score),
                          (2, gap_open_database),
                          (4, gap_open_query),
                          (3, gap_extension_database),
                          (5, gap_extension_query)]
            
            direction, max_score = max(candidates, key=lambda x: x[1])
            
            # check entry sign
            if max_score > 0:
                E_matrix[i,j] = direction
                D_matrix[i, j] = max_score
            else:
                E_matrix[i,j] = 0
                D_matrix[i, j] = 0

            # fetch global max score
            if max_score > D_matrix_max_score:
                D_matrix_max_score = max_score
                D_matrix_i_max = i
                D_matrix_j_max = j
            
    return P_matrix, Q_matrix, D_matrix, E_matrix, D_matrix_i_max, D_matrix_j_max, D_matrix_max_score


# ## Alignment Matrix Traceback

# ### Traceback

# In[18]:


def smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query="VLLP", database="VLILP", gap_open=-5, gap_extension=-1):
    
    M = len(query)
    N = len(database)
    
    aligned_query = []
    aligned_database = []
    
    # start from max_i, max_j
    i, j = i_max, j_max
    matches = 0
    while i < M and j < N:

        # E[i,j] = 0, stop back tracking
        if E_matrix[i, j] == 0:
            break
        
        
        # E[i,j] = 1, match
        if E_matrix[i, j] == 1:
            aligned_query.append(query[i])
            aligned_database.append(database[j])
            if ( query[i] == database[j]):
                matches += 1
            i += 1
            j += 1
        
        
        # E[i,j] = 2, gap opening in database
        if E_matrix[i, j] == 2:
            aligned_database.append("-")
            aligned_query.append(query[i])
            i += 1

            
        # E[i,j] = 3, gap extension in database
        if E_matrix[i, j] == 3:
                   
            count = i + 2
            score = D_matrix[count, j] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.00001):   
                count += 1
                score = D_matrix[count, j] + gap_open + (count-i-1)*gap_extension
                
            for k in range(i, count):
                aligned_database.append("-")
                aligned_query.append(query[i])
                i += 1
            
            
        # E[i,j] = 4, gap opening in query
        if E_matrix[i, j] == 4:
            aligned_query.append("-")
            aligned_database.append(database[j])
            j += 1
        
        
        # E[i,j] = 5, gap extension in query
        if E_matrix[i, j] == 5:
             
            count = j + 2
            score = D_matrix[i, count] + gap_open + gap_extension
            
            # Find length of gap
            while((score - D_matrix[i, j])*(score - D_matrix[i, j]) >= 0.0001): 
                count += 1
                score = D_matrix[i, count] + gap_open + (count-j-1)*gap_extension

            for k in range(j, count):
                aligned_query.append("-")
                aligned_database.append(database[j])
                j += 1

                
    return aligned_query, aligned_database, matches


# ## Now test the code on a few examples

# In[19]:


# Morten's slides
#query = "VLLP"
#database = "VLILP"
#scoring_scheme = blosum50
#gap_open = -5
#gap_extension = -1

#Matrix dump exercise 2
#query = "VLPVLILP"
#database = "VLLPVLLP"
scoring_scheme = blosum50
#gap_open = -2
#gap_extension = -1

#Matrix dump exercise 1
#query = "IDVLLGADDGSLAFVPSEFSISPGEKIVFKNNAGFPHNIVFDEDSIPSGVDASKISMSEEDLLNAKGETFEVALSNKGEYSFYCSPHQGAGMVGKVTVN"
#database = "AEVKLGSDDGGLVFSPSSFTVAAGEKITFKNNAGFPHNIVFDEDEVPAGVNAEKISQPEYLNGAGETYEVTLTEKGTYKFYCEPHAGAGMKGEVTVN"
#scoring_scheme = blosum50
#gap_open = -11
#gap_extension = -1

P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query, database, scoring_scheme, gap_open, gap_extension)
aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query, database, gap_open, gap_extension)
    
print("ALN", "Query", len(query), "Database", len(database), len(aligned_query), max_score, matches)
print("QAL", ''.join(aligned_query))
print("DAL", ''.join(aligned_database))

print("---")

print("D Matrix")
print(D_matrix)
print("")
print("E Matrix")
print(E_matrix)
print("")
print("P Matrix")
print(P_matrix)
print("")
print("E Matrix")
print(E_matrix)


# ## Multiple alignments

# ### Load 1PLC.tab (query)

# In[20]:


query_file = data_dir + "Align/1PLC._.tab"
#query_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/1PLC._.tab"
query_list = np.loadtxt(query_file, dtype=str).reshape(-1,2)


# ### Load database_list.tab
# 

# In[21]:


database_file = data_dir + "Align/db_100.tab"
#database_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/database_list.tab"
database_list = np.loadtxt(database_file, dtype=str).reshape(-1,2)


# ### Align query against database

# In[22]:


from time import time

scoring_scheme = blosum50
gap_open = -11
gap_extension = -1

# this returns current timestamp in seconds
t0 = time()

for query in query_list:
    
    query_protein = query[0]
    query_sequence = query[1]
    
    for database in database_list:
    
        database_protein = database[0]
        database_sequence = database[1]

        P_matrix, Q_matrix, D_matrix, E_matrix, i_max, j_max, max_score = smith_waterman_alignment(query_sequence, database_sequence, scoring_scheme, gap_open, gap_extension)
        aligned_query, aligned_database, matches = smith_waterman_traceback(E_matrix, D_matrix, i_max, j_max, query_sequence, database_sequence, gap_open, gap_extension)

        print("ALN", query_protein, len(query_sequence), database_protein, len(database_sequence), len(aligned_query), matches, max_score)
        print("QAL", i_max, ''.join(aligned_query))
        print("DAL", j_max,''.join(aligned_database))
        print("")
        
t1 = time()

print("Time (m):,",(t1-t0)/60)

