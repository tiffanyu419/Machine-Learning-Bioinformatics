"""
This program calls on three different algorithms,
C4.5 algorithm, Adaboost and Random Forest. N-fold
cross validation is used to create results from all
three algorithms.
Various Python sktlearn packages are used.
Final Project. Maggie Eberts and Tiffany Yu
"""

import numpy as np
from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from operator import itemgetter


def usage():
    import sys
    print >> sys.stderr, "Error, incorrect argument format"
    print >> sys.stderr, "Usage: python main.py input_file"
    print >> sys.stderr, "input_file - file name containing training set"

#load DataFile into an array of lists 
def load_data(input_file):
    try:
        with open(input_file, 'r') as f:
            file_list = f.readlines()
    #Invalid Data file
    except IOError:
        print "Could not read input data file...exiting program"
        exit()

    x = []
    for line in file_list:
        line = line.replace(',', ' ')
        line = line.split()
        x_list = []
        for num in line:
            x_list.append(float(num))
        x.append(x_list)

    return x

#splits data into features and class, where class is the last
#element in the point's vector
def split_data(data):
    rows = len(data)
    columns = len(data[0])
    t_x = []
    t_y = []
    for k in range(rows):
        tempx = []
        for j in range(columns):
            if j < (columns -1):
                tempx.append((data[k][j]))
            else:
                t_y.append(int(data[k][j]))
        t_x.append(tempx)

    x = np.array(t_x)
    y = np.array(t_y)
    tup = (x,y)
    return tup

#takes all data points and isolates only desired features for logistic regression
def feature_selection(data, indices):
    result = []
    for i in data:
        temp_list = []
        for j in range(10):
            index = indices[j]
            temp_list.append((i[index]))
        result.append(temp_list)

    return result

#print out accuracy metrics for each classifier
def print_results(dt_score, dt_f1, dt_recall, dt_feat, ab_score, ab_f1, ab_recall, ab_feat, rf_score, rf_f1, rf_recall, rf_feat, n):
    print("Accuracy scores from %d folds \n"% n)
    print("Decision Tree:")
    print("Accuracy Score-%.3f f1-%.3f recall-%.3f" %(dt_score, dt_f1, dt_recall))
    print("Important features:")
    print(dt_feat)
    print("\n")
    print("AdaBoost:")
    print("Accuracy Score-%.3f f1-%.3f recall-%.3f" %(ab_score, ab_f1, ab_recall))
    print("Important features:")
    print(ab_feat)
    print("\n")
    print("RandomForest:")
    print("Accuracy Score-%.3f f1-%.3f recall-%.3f" %(rf_score, rf_f1, rf_recall))
    print("Important features:")
    print(rf_feat)

def main():
    #parse command line and exit program if incorrect number of arguments given
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', action = "store")

    try:
        args = parser.parse_args()
        input_file = args.file
        data = load_data(input_file)
        num_data = len(data)

    except SystemExit:
        usage()

    # read in DataFeatures file, store in variable and varlen
    f1 = open("projectData/DataFeatures.txt", 'r')
    i=0
    features=[]
    varlen=[]
    for line in f1:
      if (i<6):
        i +=1
        continue
      if (i>104):
        break
      else:
        info = line.split()
        features.append(info[0])
        varlen.append(int(info[3]))
        i+=1
    #remove CNBRES from features list (#57)
    features = features[:57]+features[58:]

    #split data
    tup = split_data(data)
    x = tup[0]
    y = tup[1]
    #creates folds
    n = 5
    kf = KFold(num_data,5, shuffle=True, random_state=None)

    #initialize scores and feature dictionaries
    dt_score = 0
    dt_f1= 0
    dt_recall = 0
    dt_lgscore = 0
    dt_lgf1 = 0
    dt_lgrecall = 0

    ab_score = 0
    ab_lg = 0
    ab_f1= 0
    ab_recall = 0
    ab_lgscore = 0
    ab_lgf1 = 0
    ab_lgrecall = 0

    rf_score = 0
    rf_lg = 0
    rf_f1= 0
    rf_recall = 0
    rf_lgscore = 0
    rf_lgf1 = 0
    rf_lgrecall = 0

    count = 0
    k=1
    dt_feat_score= {}
    ab_feat_score= {}
    rf_feat_score= {}

    #iterate through each fold
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        """ C4.5 Decision Tree """
        # Call C4.5 decision tree
        dt = tree.DecisionTreeClassifier()
        dt.fit(x_train, y_train)
        #Calculates the metrics of trees
        dt_score += dt.score(x_test, y_test)
        pred = dt.predict(x_test)
        dt_f1 += metrics.f1_score(y_test, pred, average='binary')
        dt_recall += metrics.recall_score(y_test, pred, average='binary')
        #Getting important features
        dt_imp = dt.feature_importances_
        indices = np.argsort(dt_imp)[::-1] #sort from largest to smallest
        #save first 10 important features
        dt_feat = []
        for i in range(10):
            index = indices[i]
            dt_feat.append(features[index])
        #edit data set to only desired features
        new_Xtrain = feature_selection(x_train, indices)
        new_Xtest = feature_selection(x_test, indices)
        #train logistic model on data set with only selected features
        logistic = linear_model.LogisticRegression()
        logistic.fit(new_Xtrain, y_train)
        log_pred = logistic.predict(new_Xtest)
        dt_lgscore += logistic.score(new_Xtest, y_test)
        dt_lgf1 += metrics.f1_score(y_test, log_pred, average = 'binary')
        dt_lgrecall += metrics.recall_score(y_test, log_pred, average='binary')
        #add counts for top features to dictionary of features
        for j in range(10):
            index = indices[j]
            if dt_feat_score.has_key(index):
                prv_score = dt_feat_score[index]
                dt_feat_score[index] += 1
            else:
                dt_feat_score[index]= 1
        
        #visualize the C4.5 results
        outfile = "Results/Trial5_Out" +str(k)+".dot"
        k+=1;
        with open(outfile, 'w') as f:
            dot_data = tree.export_graphviz(dt, out_file = f, feature_names=features)

        """ AdaBoost """
        #Call AdaBoost
        ab = AdaBoostClassifier()
        ab.fit(x_train, y_train)
        #Calculates the metrics of AdaBoost
        ab_score += ab.score(x_test, y_test)
        pred = ab.predict(x_test)
        ab_f1 += metrics.f1_score(y_test, pred, average='binary')
        ab_recall += metrics.recall_score(y_test, pred, average='binary')
        #Getting important features
        ab_imp = ab.feature_importances_
        indices = np.argsort(ab_imp)[::-1] #sort from largest to smallest
        #save first 10 important features
        ab_feat = []
        for i in range(10):
            index = indices[i]
            ab_feat.append(features[index])
        #edit data set to only desired features
        new_Xtrain = feature_selection(x_train, indices)
        new_Xtest = feature_selection(x_test, indices)
        #train logistic model on data set with only selected features
        logistic = linear_model.LogisticRegression()
        logistic.fit(new_Xtrain, y_train)
        log_pred = logistic.predict(new_Xtest)
        ab_lgscore += logistic.score(new_Xtest, y_test)
        ab_lgf1 += metrics.f1_score(y_test, log_pred, average = 'binary')
        ab_lgrecall += metrics.recall_score(y_test, log_pred, average='binary')
        #add counts for top features to dict of features
        for j in range(10):
            index = indices[j]
            if ab_feat_score.has_key(index):
                prv_score = ab_feat_score[index]
                ab_feat_score[index] += 1
            else:
                ab_feat_score[index]= 1

        """ RandomForest """
        #RandomForest
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        #Calculates the metrics of AdaBoost
        rf_score += rf.score(x_test, y_test)
        pred = rf.predict(x_test)
        rf_f1 += metrics.f1_score(y_test, pred, average='binary')
        rf_recall += metrics.recall_score(y_test, pred, average='binary')
        #Getting important features
        rf_imp = rf.feature_importances_
        indices = np.argsort(rf_imp)[::-1] #sort from largest to smallest
        #save first 10 important features
        rf_feat = []
        for i in range(10):
            index = indices[i]
            rf_feat.append(features[index])
        #edit data set to only desired features
        new_Xtrain = feature_selection(x_train, indices)
        new_Xtest = feature_selection(x_test, indices)
        #train log model to data set with selected features
        logistic = linear_model.LogisticRegression()
        logistic.fit(new_Xtrain, y_train)
        log_pred = logistic.predict(new_Xtest)
        rf_lgscore += logistic.score(new_Xtest, y_test)
        rf_lgf1 += metrics.f1_score(y_test, log_pred, average = 'binary')
        rf_lgrecall += metrics.recall_score(y_test, log_pred, average='binary')
        #add counts for top features to dict of features
        for j in range(10):
            index = indices[j]
            if rf_feat_score.has_key(index):
                prv_score = rf_feat_score[index]
                rf_feat_score[index] += 1
            else:
                rf_feat_score[index]= 1

    #Average out the scores from each fold
    dt_score = dt_score/n
    dt_f1 = dt_f1/n
    dt_recall = dt_recall/n
    dt_lgscore = dt_lgscore/n
    dt_lgf1 = dt_lgf1/n
    dt_lgrecall = dt_lgrecall/n
    ab_score = ab_score/n
    ab_f1 = ab_f1/n
    ab_recall = ab_recall/n
    ab_lgscore = ab_lgscore/n
    ab_lgf1 = ab_lgf1/n
    ab_lgrecall = ab_lgrecall/n
    rf_score = rf_score/n
    rf_f1 = rf_f1/n
    rf_recall = rf_recall/n
    rf_lgscore = rf_lgscore/n
    rf_lgf1 = rf_lgf1/n
    rf_lgrecall = rf_lgrecall/n


    #Select most important features and its accuracy
    #file = to write the features for each trial
    f = open("Results/LogisticRegressionTrial5.txt", 'w')
    dt_tup = dt_feat_score.items()
    dt_tup = sorted(dt_tup,key=itemgetter(1))[::-1]
    ab_tup = ab_feat_score.items()
    ab_tup = sorted(ab_tup,key=itemgetter(1))[::-1]
    rf_tup = rf_feat_score.items()
    rf_tup = sorted(rf_tup,key=itemgetter(1))[::-1]
    f.write("Features")
    f.write("\n")
    f.write("Decision Tree:\n")
    for i in dt_tup:
        f.write("%s: %d, "%(features[i[0]],i[1]))
    f.write("\n")
    f.write("Average Accuracy Score: %.5f F1: %.5f Recall: %.5f"%(dt_lgscore, dt_lgf1, dt_lgrecall))
    f.write("\n")
    f.write("\n")
    f.write("Adaboost:\n")
    for j in ab_tup:
        f.write("%s: %d, "%(features[j[0]],j[1]))
    f.write("\n")
    f.write("Average Accuracy Score: %.5f F1: %.5f Recall: %.5f"%(ab_lgscore, ab_lgf1, ab_lgrecall))
    f.write("\n")
    f.write("\n")
    f.write("Random Forest:\n")
    for k in rf_tup:
        f.write("%s: %d, "%(features[k[0]],k[1]))
    f.write("\n")
    f.write("Average Accuracy Score: %.5f F1: %.5f Recall: %.5f\n"%(rf_lgscore, rf_lgf1, rf_lgrecall))

    #Print out results
    print_results(dt_score, dt_f1, dt_recall, dt_feat, ab_score, ab_f1, ab_recall, ab_feat, rf_score, rf_f1, rf_recall, rf_feat, n)

main()
