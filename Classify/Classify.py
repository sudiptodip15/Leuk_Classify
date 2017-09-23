''' Exploring Parametric and Non-Parametric Classifiers ''' 

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def Plot_err(param,train_err,val_err, classifier_name = 'Logistic Regression', num_classes = 18 ):
        
        plt.plot(range(len(train_err)),train_err,'bo-',label='Training')
        plt.plot(range(len(val_err)),val_err,'ro-',label='Validation')
        plt.xlabel('C')
        plt.xticks(range(len(train_err)),param)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(classifier_name + ' : ' + str(num_classes))
        plt.show()



class Classifier(object):
    
    def __init__(self,Xtr,Ytr,Xtest,Ytest):
        self.Ntr=len(Ytr)
        self.Ntest=len(Ytest)
        self.Xtr=Xtr
        self.Ytr=Ytr.reshape((self.Ntr,1))
        self.Xtest=Xtest
        self.Ytest=Ytest.reshape((self.Ntest,1))
        
     
    def SVM(self):
        
        print('SVM with Linear Kernel : One Vs Many \n')
        
        indx_v=np.linspace(start=0,stop=self.Ntr-1,num=100,dtype=int)
        indx_tr=np.setdiff1d(range(self.Ntr),indx_v)
        Xv=self.Xtr[indx_v]
        Yv=self.Ytr[indx_v]
        Xtr=self.Xtr[indx_tr]
        Ytr=self.Ytr[indx_tr]
        C_list=np.logspace(-5,2,20)
        best_C=0.01
        max_acc=0
        train_err=np.zeros((len(C_list)))
        val_err=np.zeros((len(C_list)))
        i=0
        for c in C_list:
            print('Training with C = %f' %c)
            model=LinearSVC(C=c,dual=False,random_state=0)            
            model.fit(Xtr, Ytr.reshape(1900))
            acc=model.score(Xv,Yv.reshape(100))
            train_err[i]=model.score(Xtr,Ytr.reshape(1900))
            val_err[i]=acc
            if(acc>max_acc):
                best_C=c
                max_acc=acc
        
            i+=1
            
        print('Best C = %f' %(best_C))
        model= LinearSVC(C=best_C,dual=False,random_state=0)  
        model.fit(self.Xtr, self.Ytr.reshape(self.Ntr))
        self.print_Accuracy(model)      
        
        print('Train Err :')
        print(train_err)
        print('Validation Err :')
        print(val_err)
        
        Plot_err([],train_err,val_err, 'SVM Linear')          
        
        

    def Logistic(self):
        
        print('Logistic Regression :\n')
        
        indx_v=np.linspace(start=0,stop=self.Ntr-1,num=100,dtype=int)
        indx_tr=np.setdiff1d(range(self.Ntr),indx_v)
        Xv=self.Xtr[indx_v]
        Yv=self.Ytr[indx_v]
        Xtr=self.Xtr[indx_tr]
        Ytr=self.Ytr[indx_tr]
        C_list=np.logspace(-5,2,50)
        best_C=0.01
        max_acc=0
        train_err=np.zeros((len(C_list)))
        val_err=np.zeros((len(C_list)))
        i=0
        for c in C_list:
            print('Training with C = %f' %c)
        
            model=LogisticRegression(C=c,max_iter=200,solver='sag',multi_class='multinomial')
            model.fit(self.Xtr, self.Ytr)
            model.fit(Xtr, Ytr.reshape(1900))
            acc=model.score(Xv,Yv.reshape(100))
            train_err[i]=model.score(Xtr,Ytr.reshape(1900))
            val_err[i]=acc
            if(acc>max_acc):
                best_C=c
                max_acc=acc
        
            i+=1
            
        print('Best C = %f' %(best_C))
        model= LogisticRegression(C=best_C,max_iter=200,solver='sag',multi_class='multinomial')  
        model.fit(self.Xtr, self.Ytr.reshape(self.Ntr))
        self.print_Accuracy(model)      
        
        print('Train Err :')
        print(train_err)
        print('Validation Err :')
        print(val_err)
        
        Plot_err([],train_err,val_err,'Logistic Regression')      
       
        
    def DTree(self):
		
		# Prone to overfitting in sparse data regime. Hence, not much experimentation done. Fit with default values.
        
        print('Decision Tree :\n')
        model=DecisionTreeClassifier(random_state=0)
        model.fit(self.Xtr, self.Ytr)        
        self.print_Accuracy(model)        
        
    def kNN(self):
        
        print('k-Nearest Neighbor :\n')
        
        indx_v=np.linspace(start=0,stop=self.Ntr-1,num=100,dtype=int)
        indx_tr=np.setdiff1d(range(self.Ntr),indx_v)
        Xv=self.Xtr[indx_v]
        Yv=self.Ytr[indx_v]
        Xtr=self.Xtr[indx_tr]
        Ytr=self.Ytr[indx_tr]
        k_list=np.arange(1,51)
        best_k=-1
        max_acc=0
        train_err=np.zeros((len(k_list)))
        val_err=np.zeros((len(k_list)))
        for k in k_list:
            print('Training with k = %d' %k)
            model= KNeighborsClassifier(n_neighbors=k)
            model.fit(Xtr, Ytr.reshape(1900))
            acc=model.score(Xv,Yv.reshape(100))
            train_err[k-1]=model.score(Xtr,Ytr.reshape(1900))
            val_err[k-1]=acc
            if(acc>max_acc):
                best_k=k
                max_acc=acc
        
        
        print('Best k = %d' %(best_k))
        model= KNeighborsClassifier(n_neighbors=best_k)
        model.fit(self.Xtr, self.Ytr.reshape(self.Ntr))
        self.print_Accuracy(model)     
        
       
        
        Plot_err(k_list,train_err,val_err, ' kNN classifier')
        
        
        
    def print_Accuracy(self,model):
         
        tr_acc=model.score(self.Xtr,self.Ytr)
        test_acc=model.score(self.Xtest,self.Ytest)
        
        print('Train Accuracy = %f' %tr_acc)
        print('Test Accuracy = %f' %test_acc)  
      
      
      
      
      
if __name__=='__main__':


    
    Xtr = np.loadtxt('Train_Data.txt')
    Ytr = np.loadtxt('Train_Label.txt')
    Xtest = np.loadtxt('Test_Data.txt')
    Ytest = np.loadtxt('Test_Label.txt')
   
    
    print('Data Loaded')    
    
    
    Leuk=Classifier(Xtr,Ytr,Xtest,Ytest)
    Leuk.SVM()
    Leuk.Logistic()
    Leuk.DTree()
    Leuk.kNN()




