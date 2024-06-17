#!/usr/bin/env python
# coding: utf-8

# # SMM with Gradient Descent

# ## Python Imports

# In[106]:


import numpy as np
import random
import copy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Imports

# ## DEFINE THE PATH TO YOUR COURSE DIRECTORY
#get_ipython().run_line_magic('matplotlib', 'inline')
from argparse import ArgumentParser

# We will now create an argument parser that will receive as arguments two values
# 1) the peptides-targets file to open (-f option)
# 2) the threshold to be applied in the target value filtering step (-t option)
# To achieve this, add the following lines below the ArgumentParser import line:


parser = ArgumentParser(description="SMM gradient descent")
parser.add_argument("-l", action="store", dest="LAMB", type=float, default = 0.01, help="Lambda (default: 0.01)")
parser.add_argument("-t", action="store", dest="TRAINING_FILE", type=str, help="File with training data")
parser.add_argument("-e", action="store", dest="EVALUATION_FILE", type=str, help="File with evalutation data")
parser.add_argument("-epi", action="store", dest="EPSILON", type=float, default=0.05, help="Epsilon (default 0.05)")
parser.add_argument("-seed", action="store", dest="SEED", type=int, default = 1, help="Seed for random numbers (default 1)")
parser.add_argument("-i", action="store", dest="EPOCHS", type=int, default = 100, help="Number of epochs to train (default 100)")
args = parser.parse_args()
lamb = args.LAMB
training_file = args.TRAINING_FILE
evaluation_file = args.EVALUATION_FILE
epsilon = args.EPSILON
seed = args.SEED
epochs = args.EPOCHS



# In[107]:


data_dir = "/home/anton_ws/Algo/data/"


# ### Training Data

# In[108]:


#training_file = data_dir + "SMM/A0201_training"
#training_file = data_dir + "SMM/A2403_training"

training = np.loadtxt(training_file, dtype=str)


# ### Evaluation Data

# In[109]:


#evaluation_file = data_dir + "SMM/A0201_evaluation"
#evaluation_file = data_dir + "SMM/A2403_evaluation"
evaluation = np.loadtxt(evaluation_file, dtype=str)


# ### Alphabet

# In[110]:


alphabet_file = data_dir + "Matrices/alphabet"
alphabet = np.loadtxt(alphabet_file, dtype=str)


# ### Sparse Encoding Scheme

# In[111]:


sparse_file = data_dir + "Matrices/sparse"
_sparse = np.loadtxt(sparse_file, dtype=float)
sparse = {}

for i, letter_1 in enumerate(alphabet):
    
    sparse[letter_1] = {}

    for j, letter_2 in enumerate(alphabet):
        
        sparse[letter_1][letter_2] = _sparse[i, j]


# ## Peptide Encoding

# In[112]:


def encode(peptides, encoding_scheme, alphabet):
    
    encoded_peptides = []

    for peptide in peptides:

        encoded_peptide = []

        for peptide_letter in peptide:

            for alphabet_letter in alphabet:

                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])

        encoded_peptides.append(encoded_peptide)
        
    return np.array(encoded_peptides)


# ## Error Function

# In[113]:


def cumulative_error(peptides, y, lamb, weights):

    error = 0
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]

        # get target prediction value
        y_target = y[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
            
        # calculate error
        error += 1.0/2 * (y_pred - y_target)**2
        
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
        
    return gerror, error


# ## Predict value for a peptide list

# In[114]:


def predict(peptides, weights):

    pred = []
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
        
        pred.append(y_pred)
        
    return pred


# ## Calculate MSE between two vectors

# In[115]:


def cal_mse(vec1, vec2):
    
    mse = 0
    
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
        
    mse /= len(vec1)
    
    return( mse)


# ## Gradient Descent

# In[116]:


def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    
    # do is dE/dO
    do = y_pred -y_target
        
    for i in range(0, len(weights)):
        
        de_dw_i = do*peptide[i] + 2*lamb_N*weights[i]

        weights[i] -= epsilon*de_dw_i


# ## Main Loop
# 
# 

# In[117]:


# Random seed 
np.random.seed( seed )

# peptides
peptides = training[:, 0]
peptides = encode(peptides, sparse, alphabet)
N = len(peptides)

# target values
y = np.array(training[:, 1], dtype=float)

#evaluation peptides
evaluation_peptides = evaluation[:, 0]
evaluation_peptides = encode(evaluation_peptides, sparse, alphabet)

#evaluation targets
evaluation_targets = np.array(evaluation[:, 1], dtype=float)

# weights
input_dim  = len(peptides[0])
output_dim = 1
w_bound = 0.1
weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

# training epochs
#epochs = 100

# regularization lambda
#lamb = 1
#lamb = 10
#lamb = 0.01

# regularization lambda per target value
lamb_N = lamb/N

# learning rate
#epsilon = 0.005

# error  plot
gerror_plot = []
mse_plot = []
train_mse_plot = []
eval_mse_plot = []
train_pcc_plot = []
eval_pcc_plot = []

# for each training epoch
for e in range(0, epochs):

    # for each peptide
    for i in range(0, N):

        # random index
        ix = np.random.randint(0, N)
        
        # get peptide       
        peptide = peptides[ix]

        # get target prediction value
        y_target = y[ix]
       
        # get initial prediction
        y_pred = np.dot(peptide, weights)

        # gradient descent 
        gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)

    # compute error
    gerr, mse = cumulative_error(peptides, y, lamb, weights) 
    gerror_plot.append(gerr)
    mse_plot.append(mse)
    
    # predict on training data
    train_pred = predict( peptides, weights )
    train_mse = cal_mse( y, train_pred )
    train_mse_plot.append(train_mse)
    train_pcc = pearsonr( y, train_pred )
    train_pcc_plot.append( train_pcc[0] )
        
    # predict on evaluation data
    eval_pred = predict(evaluation_peptides, weights )
    eval_mse = cal_mse(evaluation_targets, eval_pred )
    eval_mse_plot.append(eval_mse)
    eval_pcc = pearsonr(evaluation_targets, eval_pred)
    eval_pcc_plot.append( eval_pcc[0] )
    
    #print ("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)


# ## Error Plot

# In[118]:


#fig = plt.figure(figsize=(10, 10), dpi= 80)
#
#x = np.arange(0, len(gerror_plot))
#
#plt.subplot(2, 2, 1)
#plt.plot(x, gerror_plot)
#plt.ylabel("Global Error", fontsize=10);
#plt.xlabel("Iterations", fontsize=10);
#
#plt.subplot(2, 2, 2)
#plt.plot(x, mse_plot)
#plt.ylabel("MSE", fontsize=10);
#plt.xlabel("Iterations", fontsize=10);
#
#
#x = np.arange(0, len(train_mse_plot))
#
#plt.subplot(2, 2, 3)
#plt.plot(x, train_mse_plot, label="Training Set")
#plt.plot(x, eval_mse_plot, label="Evaluation Set")
#plt.ylabel("Mean Squared Error", fontsize=10);
#plt.xlabel("Iterations", fontsize=10);
#plt.legend(loc='upper right');
#
#
#plt.subplot(2, 2, 4)
#plt.plot(x, train_pcc_plot, label="Training Set")
#plt.plot(x, eval_pcc_plot, label="Evaluation Set")
#plt.ylabel("Pearson Correlation", fontsize=10);
#plt.xlabel("Iterations", fontsize=10);
#plt.legend(loc='upper left');


# ## Get PSSM Matrix

# ### Vector to Matrix

# In[119]:


# our matrices are vectors of dictionaries
def vector_to_matrix(vector, alphabet):
    
    rows = int(len(vector)/len(alphabet))
    
    matrix = [0] * rows
    
    offset = 0
    
    for i in range(0, rows):
        
        matrix[i] = {}
        
        for j in range(0, 20):
            
            matrix[i][alphabet[j]] = vector[j+offset] 
        
        offset += len(alphabet)

    return matrix


# ### Matrix to Psi-Blast

# In[120]:


def to_psi_blast(matrix):

    # print to user
    
    header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

    letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    for i, row in enumerate(matrix):

        scores = []

        scores.append(str(i+1) + " A")

        for letter in letter_order:

            score = row[letter]

            scores.append(round(score, 4))

        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 


# ### Print

# In[121]:


matrix = vector_to_matrix(weights, alphabet)
to_psi_blast(matrix)


# ## Performance Evaluation

# In[122]:


evaluation_peptides = evaluation[:, 0]
evaluation_peptides = np.array(encode(evaluation_peptides, sparse, alphabet))

evaluation_targets = np.array(evaluation[:, 1], dtype=float)

y_pred = []
for i in range(0, len(evaluation_peptides)):
    y_pred.append(np.dot(evaluation_peptides[i].T, weights))

y_pred = np.array(y_pred)


# In[123]:


#pcc = pearsonr(evaluation_targets, np.array(y_pred))
#print("PCC: ", pcc[0])
#
#plt.scatter(y_pred, evaluation_targets);


# In[ ]:





# In[ ]:




