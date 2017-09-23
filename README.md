## Synopsis

Leukemia is a type of blood cancer that has several sub-types and early determination of the sub-type can lead to effective cancer treatment. It is a supervised learning problem that explores various 
well known classifiers operating in a sparse data region. The challenges in this dataset are the availability of only a few thousand patient samples. Labeled data is costly in biological applications. 
Moreover, there is imbalance in class labels in the dataset. 

## Motivation

In this project, we study the various classifiers and evaluate their performance in supervised classification. There is an additional step in this project which uses Graphical Lasso for identifying possible 
gene interactions in leukemia. By suitably adjusting the sparsity of the undirected graph, the underlying structure can be detected and important links between genes determined.


## Installation

- Download the git folder and run `download_data.sh` bash script. This will download the data in the folder `Data`.
- cd Classify and run `Run_Classify.sh` bash script to generate test-train split, data cleaning and running classifiers. This will also generate the plots. Then cd ..
- cd Graphical_Lasso and run `Run_GLasso.sh` bash script to extract a subset of genes from cancer patients. You can modify the subset by changing few lines of code in `Cancer_DataGen.py`. 
Also I have provided a helper code to choose a reasonable sparsity level in Graphical Lasso based on output number of connected componets in graph 


# Performance 

In PCA reduced data-set, Logistic Regression and SVM had performed best on this task. It would be interesting to also explore random forests, xgboost and mixture of experts on this problem.  


 

