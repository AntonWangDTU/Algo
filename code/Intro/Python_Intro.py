#!/usr/bin/env python
import numpy as np

################################
# ADDING A COMMAND LINE PARSER #
################################

# For this step, we need first to import an argument parser
# to do this, add the following line just below the numpy import:

from argparse import ArgumentParser

# We will now create an argument parser that will receive as arguments two values
# 1) the peptides-targets file to open (-f option)
# 2) the threshold to be applied in the target value filtering step (-t option)
# To achieve this, add the following lines below the ArgumentParser import line:


parser = ArgumentParser(description="A very first Python program")
parser.add_argument("-t", action="store", dest="threshold", type=float, default=0.5, help="Target value filtering threshold (default: 0.5)")
parser.add_argument("-f", action="store", dest="peptides_targets_file", type=str, help="Peptides-Targets file")
parser.add_argument("-l", action="store", dest="length", type=int, default = 9, help="Kmer filter length(defeault=9)")
args = parser.parse_args()
threshold = args.threshold
length = args.length
peptides_targets_file = args.peptides_targets_file


#After adding these lines, you will now be able to call this python program 
#from the terminal while specifying these arguments:

# python Python_Intro.py -t some_threshold -f file_with_peptides_and_targets

# Note you can also parse switches with the ArgumentParser, i.e 
# parser.add_argument('-w', action='store_true', default=False, dest='sequence_weighting', help='Use sequence weighting')


# REMEMBER!
# 1) The argument parser needs to be declared on the beginning of the script, right after the imports
# 2) In order for this program to work properly after adding the parser, you must now comment or delete 
#    the previous declarations of the variables "threshold" and "peptides_target_file"

# ## DEFINE THE PATH TO YOUR COURSE DATA DIRECTORY
data_dir = "../../data/"


# ## Load peptides-targets data

# **Specify file path**
#peptides_targets_file = data_dir + "Intro/test.dat"


# **Load the file with the numpy text parser *np.loadtxt* and reshape it into a numpy array of shape (-1, 2)** 
peptides_targets = np.loadtxt(peptides_targets_file, dtype=object)

print(peptides_targets.shape)


# ## Store peptides in vector
peptides = peptides_targets[:, 0]

print(type(peptides), type(peptides_targets))


# ## Store targets in vector

# **Fish out the target values from the loaded file using array indexing and slicing**
targets = peptides_targets[:, 1].astype(float)


# ## Keep 9mers only

peptides_9mer = []
targets_9mer = []


# **Iterate over the elements of the peptides list and keep peptides with length == 9 using the .append command**
for i in range(0, len(peptides)):
    
    if len(peptides[i]) == length:
        
        peptides_9mer.append(peptides[i])
        
        targets_9mer.append(targets[i])


# ## Remove peptides with target value < threshold

# **Declare a threshold variable**

# In[50]:


#threshold = 0.5


# **Declare python list to store the indexes of the elements to be removed**

# In[51]:


to_remove = []


# **Iterate over the 9mer peptides, check which target values < threshold, and store the indexes in the to_remove  array**

# In[52]:


for i in range(0, len(peptides_9mer)):
        
        if targets_9mer[i] < threshold:

            to_remove.append(i)


# **Use the *delete* NumPy function to remove the peptides**

# In[53]:


peptides_9mer_t = np.delete(peptides_9mer, to_remove)
targets_9mer_t = np.delete(targets_9mer, to_remove)


# **Check that no elements with target < threshold are present in the target values array**

# In[54]:


error = False

for i in range(0, len(peptides_9mer_t)):
        
        if targets_9mer_t[i] < threshold:

            error = True
            
            break

if error:

    print("Something went wrong")
    
else:
    
    print("Success")


# ## Print the final, filtered peptide-target pairs

# **Ensure that this output is consistent with the data filtering steps you have made!**

# In[55]:


for i in range(0, len(peptides_9mer_t)):
    
    print(peptides_9mer_t[i], targets_9mer_t[i])






