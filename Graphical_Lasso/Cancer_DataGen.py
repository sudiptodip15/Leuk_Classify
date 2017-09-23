''' Generate a Dataset of Subset of Genes from all Cancer Patients '''

import numpy as np
count=0
f=open('../Data/Label.txt','r')

index_cancer = []
while(True):
    line=f.readline().rstrip().split('\t')
    if(line[1]=='Non-leukemia and healthy bone marrow'):
        break
    else:        
        index_cancer.append(count)
        count+=1
f.close()
print('Count = '+str(count))

X = np.genfromtxt("mile_cleaned.csv", delimiter=",")
# Extract a subset of genes for all cancer patients 
subset_size = 500
gene_subset  = np.arange(subset_size)
X=X[gen_subset,index_cancer].T
np.savetxt('Cancer_Data.txt',X)


