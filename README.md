# Comparing Performance of Machine Learning Algorithms through deterimining Breast Cancer Risk Factors
by Tiffany Yu and Maggie Eberts

Through determining the risk factors of breast cancer using three different Machine Learning
Algorithms - C4.5, AdaBoost and RandomForest, we aim to evaluate and contrast the perfomance
of these algorithms. Running these algorithms on the same data set, we looked at the accuracy, 
specificity and sensitivity of the predictions of these algorithms. The results were internally
validated using logistic regression and externally validated through the research of risk factors
in official medical journals. 

To implement the C4.5, AdaBoost and RandomForest algorithms, 
packages from sklearn were used.

1. C4.5: from sklearn.tree import DecisionTreeClassifier
2. AdaBoost: from sklearn.ensemble import AdaBoostClassifier
3. RandomForest: from sklearn.ensemble import RandomForestClassifier

#### modifyData.py:
  This python programs takes the raw data file (Cancer.dat), modifies the data as
  needed for experiments, and outputs two data files (ModCancerData.csv - the 
  complete data set used for experiments and debug.csv - a smaller data set used
  for debugging purposes)

#### main.py:
  This python program includes the entire code to run experiments. To run 
  experiments, run "python main.py [datatextfile]".  For our experiments,
  ModCancerData.csv was used as the [datatextfile].

#### projectData Folder:
  Cancer.dat - raw data file from IPUMS database
  DataFeatures.txt - File containing information on features of the IPUMS
    data set
