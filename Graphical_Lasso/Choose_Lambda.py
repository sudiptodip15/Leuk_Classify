''' Vary parameter lambda and Check the number of connected components '''

import numpy as np
import scipy.sparse.csgraph as sg

X=np.loadtxt('Cancer_Data.txt')

N,d=X.shape
mu=np.mean(X,axis=0)
sig=np.std(X,axis=0)

X=(X-mu)/sig

C=(1.0/N)*np.dot(X.T,X)

f=open('lamdba_sparse.txt','w')
for lmda in np.linspace(0,1,100):
    Adj=C.copy()
    Adj[abs(Adj)<lmda]=0
    Adj[abs(Adj)>=lmda]=1
    
    n_components,labels=sg.connected_components(Adj,directed=False)
    
    unique, counts = np.unique(labels,return_counts=True)
    
    size_max=np.max(counts)
    num_large=np.sum(counts>1)
    
    print('Lambda = %f,  n_components = %d,  size_max = %d' %(lmda,num_large, size_max) )
    
    f.write(str(lmda)+"\t"+str(num_large)+"\t"+str(size_max)+"\n")

f.close()
    
