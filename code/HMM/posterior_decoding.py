#!/usr/bin/env python
# coding: utf-8

# # Posterior Decoding
# Implementation of the Posterior Decoding algorithm.
# 
# 

# ## Python Imports


import numpy as np


#Command parser
from argparse import ArgumentParser

parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-i", action="store", dest="INPUT_FILE", type=str, help="Input File")
parser.add_argument("-s", action="store", dest="STATE", type=int, help="State")
args = parser.parse_args()
input_file = args.INPUT_FILE
p_state = args.STATE


with open(input_file, 'r') as f:
    input_sequence = f.read().strip()

print(input_sequence)


# ## Encode sequence as integers (index values)


# In[42]:


def encode( sequence, symbols):
    
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)


# ## Parameters

# In[43]:


states = 2

symbols = "123456"

#input_sequence = "566611234"
#input_sequence = "31245366664"
#input_sequence = "34512331245366664666563266"

input_encode = encode( input_sequence, symbols)

initial_prob = [1.0/states, 1.0/states]

transition_matrix = np.asarray([0.95, 0.05, 0.1, 0.9]).reshape(2,2)

fair_prob = [1.0/6, 1./6, 1./6, 1./6, 1./6, 1./6]
loaded_prob = [1./10, 1./10, 1./10, 1./10, 1./10, 5./10] 
emission_probs = [fair_prob, loaded_prob]


# ## Forward Loop
# ### Remember that we here do NOT work in log space

# In[44]:


def initialize_forward(input_encode, states, initial_prob, emission_probs):
    
    alpha = np.zeros(shape=(states, len(input_encode)))
        
    for i in range(0, states): 
        
        alpha[i][0] = initial_prob[i]*emission_probs[i][input_encode[0]]
        
    return alpha


# In[45]:


alpha = initialize_forward(input_encode, states, initial_prob, emission_probs)

# main loop
for i in range(1, len(input_encode)):
    
    for j in range(0, states):

        _sum = 0
        
        for k in range(0, states):
            
            _sum += alpha[k][i-1]*transition_matrix[k][j]   #XX           
         
        # store prob
        alpha[j][i] = emission_probs[j][input_encode[i]] * _sum


# ## Backward Loop

# In[46]:


def initialize_backward(input_encode, states):
    
    #beta = np.zeros(shape=(states, len(input_encode), dtype=float))
    beta = np.zeros(shape=(states, len(input_encode)))
        
    for i in range(0, states):
  
        beta[i][-1] = 1
        
    return beta


# In[47]:


beta = initialize_backward(input_encode, states)

# main loop
for i in range(len(input_encode)-2, -1, -1):
    
    for j in range(0, states):

        _sum = 0

        for k in range(0, states):
            
            _sum +=  transition_matrix[j][k] * beta[k][i+1] * emission_probs[k][input_encode[i+1]] #
        
         #store prob
        beta[j][i] = _sum  


# ## Posterior Loop
#  

# In[48]:


# posterior = f * b / p_x

posterior = np.zeros(shape=(len(input_encode)), dtype=float)

#p_state = 0

p_x = 0
for j in range(0, states):
    p_x += alpha[j][-1]

print ("Log(Px):", np.log(p_x),"aka Sum of the state rows XD") 

for i in range(0, len(input_encode)):
        
    posterior[i] = (alpha[p_state][i]*beta[p_state][i])/p_x # p = (f_i * b_i)/p_x

    print ("Posterior", i, input_sequence[i], input_encode[i], np.log(alpha[p_state, i]), np.log(beta[p_state, i]), posterior[i])


# In[ ]:




