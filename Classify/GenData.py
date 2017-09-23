''' Converts text labels into integer classes and does train-test split of data '''

import numpy as np

X = np.genfromtxt("../Data/mile_cleaned.csv", delimiter=",")
X=X[1:,1:].T

dic_label={'ALL with hyperdiploid karyotype': 1,
 'ALL with t(12;21)': 1,
 'ALL with t(1;19)': 1,
 'AML complex aberrant karyotype': 3,
 'AML with inv(16)/t(16;16)': 3,
 'AML with normal karyotype + other abnormalities': 3,
 'AML with t(11q23)/MLL': 3,
 'AML with t(15;17)': 3,
 'AML with t(8;21)': 3,
 'CLL': 2,
 'CML': 5,
 'MDS': 4,
 'Non-leukemia and healthy bone marrow': 0,
 'Pro-B-ALL with t(11q23)/MLL': 1,
 'T-ALL': 1,
 'c-ALL/Pre-B-ALL with t(9;22)': 1,
 'c-ALL/Pre-B-ALL without t(9;22)': 1,
 'mature B-ALL with t(8;14)': 1}

Ntest=93
N,G=X.shape
indx_test=np.linspace(start=0,stop=N-1,num=93,dtype=int)
indx_tr=np.setdiff1d(range(N),indx_test)
Y=np.ones((N,1))*np.inf             

f=open('../Data/Label.txt','r')
k=0
while(True):
    line=f.readline().rstrip()
    if(len(line)==0):
        break
    line=line.split('\t')
    Y[k]=dic_label[line[1]]
    k+=1
f.close()         


Xtr=X[indx_tr]             
Ytr=Y[indx_tr]
Xtest=X[indx_test]
Ytest=Y[indx_test]


np.savetxt('Train_Data.txt',Xtr,fmt='%f')
np.savetxt('Train_Label.txt',Ytr,fmt='%d')
np.savetxt('Test_Data.txt',Xtest,fmt='%f')
np.savetxt('Test_Label.txt',Ytest,fmt='%d')
