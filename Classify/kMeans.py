''' Unsupervised Classifier for Leukemia Classification using kmeans '''

from sklearn.cluster import KMeans
import numpy as np
Xtr = np.loadtxt('Train_Data.txt')
Ytr = np.loadtxt('Train_Label.txt')
Xtest = np.loadtxt('Test_Data.txt')
Ytest = np.loadtxt('Test_Label.txt')


Ntr,_=Xtr.shape
Ntest,_=Xtest.shape

cl_list=np.linspace(6,200,10,dtype=int)

iter_count=0
Tr_array=np.ones(10)*np.inf
Test_array=np.ones(10)*np.inf

for num_cl in cl_list:
    
    kmeans = KMeans(n_clusters=num_cl, random_state=0, max_iter=50).fit(Xtr)
    Tr_cluster=kmeans.labels_
    class_mu=np.ones(num_cl)*(-1)
    errTr=0
    for i in range(num_cl):
          indx=np.where(Tr_cluster==i)[0]  
          vote=Ytr[indx]
          unique,counts=np.unique(vote,return_counts=True)  
          class_mu[i]=unique[int(np.argmax(counts))]
          errTr+=(np.sum(counts)-np.max(counts))
            
    Test_cluster=kmeans.predict(Xtest)
    Yhat_test=class_mu[Test_cluster].reshape(Ntest,1)
    errTest=np.sum(Yhat_test!=Ytest)
    
    print('\nNumber of Cluster = %d' %(num_cl))
    print('Train 0/1 Loss = %f' %(errTr*1.0/Ntr))
    print('Test 0/1 Loss = %f' %(errTest*1.0/Ntest))
    
    Tr_array[iter_count]=errTr*1.0/Ntr
    Test_array[iter_count]=errTest*1.0/Ntest
    iter_count+=1

np.savetxt('KMeansErrTrain.txt',Tr_array)
np.savetxt('KMeansErrTest.txt',Test_array)
    
