''' Apply Graphical Lasso on the Connected Component Groups.  '''

import numpy as np
import scipy.sparse.csgraph as sg
from sklearn.covariance import GraphLassoCV as gl
X=np.loadtxt('Cancer_Data.txt')

N,d=X.shape
mu=np.mean(X,axis=0)
sig=np.std(X,axis=0)

print('Data loaded and Normalized ...\n')


X=(X-mu)/sig

C=(1.0/N)*np.dot(X.T,X)

print('Covariance Computed ...\n')

lmda_list=[0.85, 0.90, 0.95]
for lmda in lmda_list:
    print('Lambda = %f \n' %(lmda))
    Adj=C.copy()
    Adj[abs(Adj)<lmda]=0
    Adj[abs(Adj)>=lmda]=1
    
    n_components,labels=sg.connected_components(Adj,directed=False)
    
    unique, counts = np.unique(labels,return_counts=True)
    
    size_max=np.max(counts)
    num_large=np.sum(counts>3)
    indx=np.where(labels==unique[np.argmax(counts)])[0]
    S_max=C[indx][:,indx]
    
    model=gl(alphas=100,max_iter=1000)
    model.fit(X[:,indx])
    
    Theta=model.precision_
    
    fnameInvCov='InvCov_'+str(lmda)+'.txt'    
    np.savetxt(fnameInvCov,Theta,fmt='%d')
    
    fnameGene='Genes_'+str(lmda)+'.txt'
    np.savetxt(fnameGene,indx,fmt='%d')






