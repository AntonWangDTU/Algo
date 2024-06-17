#!/usr/bin/env python
# coding: utf-8

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

with open(input_file, 'r') as f:
    input_sequence = f.read().strip()

print(input_sequence)


# In[68]:


def initialize(encode_sequence, states, initial_prob, transition_matrix, emission_probs):
    
    delta = np.zeros(shape=(states, len(encode_sequence)))
    
    arrows = np.ndarray(shape=(states, len(encode_sequence)), dtype=object)
    
    # initial conditions
    for i in range(0, states):
  
        delta[i][0] = initial_prob[i] * emission_probs[i][encode_sequence[0]] # Remember we work in log space 
    
        arrows[i][0] = 0
    
    return delta, arrows


# ### Encode sequence as integers (index values)

# In[69]:


# Thies encodes the sequence to be indexes according to the symbols = "123456"

def encode( sequence, symbols):
    
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)


# ### Main Loop
#  

# In[70]:


states = 2

symbols = "123456"
nsymbols = len(symbols)

#input_sequence = "566611234"
#input_sequence = "31245366664"
#input_sequence = "34512331245366664666563266"
#input_sequence = "566611111111112346666666"
input_encode = encode( input_sequence, symbols) 

# Define model - Note this is done in log space
initial_prob = np.log10([1.0/states, 1.0/states])

transition_matrix = np.log10(np.asarray([0.95, 0.05, 0.1, 0.9]).reshape(2,2))

fair_prob = np.log10([1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6]) 
loaded_prob = np.log10([1.0/10, 1.0/10, 1.0/10, 1.0/10, 1.0/10, 5.0/10])  
emission_probs = [fair_prob, loaded_prob]

delta, arrows = initialize(input_encode, states, initial_prob, transition_matrix, emission_probs)

# main loop
for i in range(1, len(input_sequence)):
    
    for j in range(0, states):
        
        max_arrow_prob = -np.inf # A very low negative number
        max_arrow_prob_state = -1
        
        for k in range(0, states):
            
            # arrow_prob is the probability of ending in the state j from the state k
            arrow_prob = delta[k][i-1] + transition_matrix[k][j]  #XX
            
            if arrow_prob > max_arrow_prob: 
                max_arrow_prob = arrow_prob
                max_arrow_prob_state = k 
            
        # store prob
        delta[j][i] = emission_probs[j][input_encode[i]] + max_arrow_prob #XX

        # store arrow
        arrows[j][i] = max_arrow_prob_state


# ### Backtracking

# In[71]:


path = []

max_state = np.argmax(delta[:, -1]) # Find the index of the max value in the last column of delta
max_value = delta[max_state, -1] # Find the max value in the last column of delta

print("log(Max_path):", max_value)

print("Seq: ", input_sequence)

path.append(str(max_state))

old_state = max_state

for i in range(len(input_encode)-2, -1, -1):
    
    current_state = arrows[old_state][i+1] #XX
    
    path.append(str(current_state))

    old_state = current_state 
    
print("Path:", "".join(reversed(path)))

