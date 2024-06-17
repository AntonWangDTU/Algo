#!/usr/bin/env python
# coding: utf-8

# # Hobohm 2
# 
# ### Fill in the blanked out parts of the code (XX's)

# ## Python Imports

# In[1]:


import numpy as np
from time import time


# ## Data Imports
from argparse import ArgumentParser

# ## Data Imports

parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-f", action="store", dest="alignment_file", type=str, help="File input data")
args = parser.parse_args()
alignment_file_part = args.alignment_file
# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY

# In[2]:


data_dir = "../../data/"


# In[14]:


# File with formated similarity scores
#alignment_file_part = data_dir + "Hobohm/alignment_aln.fmt"

alignment_output = np.loadtxt(alignment_file_part, dtype=str)


# ## Get list of unique ID's in alignment file

# In[15]:


def get_IDlist(alignment_output):

    IDlist = []
    first = 1
    
    for row in alignment_output:
        
        # The first entry is only included as query
        if first:
            id = row[0]
            first = 0
            if id not in IDlist:
                IDlist.append(id)
            
        id = row[1]
    
        if id not in IDlist:
            IDlist.append(id)
        
    return( IDlist )


# ## Main code

# In[16]:


t0 = time()

ID_list = get_IDlist(alignment_output)

nid_list = len(ID_list)

print ("NID:", nid_list)
neighbor_matrix = np.zeros(shape=(nid_list, nid_list))

for i in range(nid_list):
    neighbor_matrix[i][i] = 1

for row in alignment_output:
        
    query_id = row[0]
    database_id = row[1]
    match = row[2]
    
    ix = ID_list.index( query_id )
    iy = ID_list.index( database_id )
    
    neighbor_matrix[ix][iy] = match
    neighbor_matrix[iy][ix] = match
    
used = np.zeros(nid_list)

left = 1

while left > 0:
    max_nn = -99;
    n_max = -9
    
    for i in range(nid_list):

        if used[i] == 0:
            
            nn = 0
            
            for j in range(nid_list):
                if used[j] == 0 and neighbor_matrix[i][j] == 1:
                    nn += 1
            
            if nn > max_nn:
                max_nn = nn
                n_max = i

    print ("# Remove", max_nn, n_max, ID_list[n_max])
    if max_nn == 1:
        left = 0
    else:
        used[n_max] = 1

t1 = time()
print ("Elapsed time (m):", (t1-t0)/60)
        
ncl = 0;
for i in range(nid_list):
    if used[i] == 0: 
        print  ("Unique", ID_list[i], ncl)
        ncl += 1
        
print ("Number of unique sequences:", ncl)


# In[ ]:





# In[ ]:




