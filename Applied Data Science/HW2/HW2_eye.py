# coding: utf-8
# NOTE THIS SCRIPT IS WRITTEN FOR python 3.6

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt

# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension

# this if for printing the confusion matrix (feel free to modify the function)
# Assumption: "pred" (data with predicted value + actual values for each observation) is a list
def print_cont_table(pred, cutoff=0.5):

    # converting pandas dataframe to list
    # If using list comment below
    pred_data = pred.iloc[:, 0:2].copy()
    data_output = pd.np.array(pred_data)
    # If using list comment above

    predicted_output = [float(i[0]) for i in data_output] # first column of list are the Predicted values
    actual_output = [int(i[1]) for i in data_output] # Second column of list are the Actual values
    # returns list = [[FALSE, TRUE][IF CONDITION] FOR a row in LIST]
    predicted_output_manipulate = [[0, 1][x > cutoff] for x in predicted_output] # can also be performed by map(lambda x: [0, 1][x > cutoff], predicted_output)

    n11_TP = sum([[0, 1][predicted_output_manipulate[i] == 1 and actual_output[i] == 1] for i in range(len(predicted_output))])
    n00_TN = sum([[0, 1][predicted_output_manipulate[i] == 0 and actual_output[i] == 0] for i in range(len(predicted_output))])
    n10_FN = sum([[0, 1][predicted_output_manipulate[i] == 0 and actual_output[i] == 1] for i in range(len(predicted_output))])
    n01_FP = sum([[0, 1][predicted_output_manipulate[i] == 1 and actual_output[i] == 0] for i in range(len(predicted_output))])
    Pos = n11_TP + n10_FN
    Neg = n01_FP + n00_TN
    PPos = n11_TP + n01_FP
    PNeg = n10_FN + n00_TN
    print ("           |  PPos \t PNeg \t | Sums")
    print ("-------------------------------------")
    print ("actual pos |  %d \t %d \t | %d" % (n11_TP, n10_FN, Pos))
    print ("actual neg |  %d \t %d \t | %d" % (n01_FP, n00_TN, Neg))
    print ("-------------------------------------")
    print ("Sums       |  %d \t %d \t | %d" % (PPos, PNeg, (Pos+Neg)))
    return None


##############################
# PART 1
print ('\n ############ PART 1 ############# \n')
##############################

# Logistic Regression
# Assumes the last column of data is the output dimension
def get_pred_logreg(train,test):
    # Your implementation goes here
    # You may leverage the linear_model module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    n1, m1 = train.shape  # number of rows and columns
    train_input = train.iloc[:,:m1-1]
    train_output = train.iloc[:,m1-1]

    n2, m2 = test.shape
    test_input = test.iloc[:,:m2-1]
    test_output = test.iloc[:,m2-1]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(train_input, train_output)

    pred = regr.predict(test_input) #predicted output
    actual = test_output #actual output
    pred = pd.DataFrame(pred)
    # reset indexes just in case
    actual = actual.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    output = pd.concat([pred, actual], axis=1)#combine panda frames
    output.columns = ['pred','actual']#add labels to columns
    return output

# Support Vector Machine
# Assumes the last column of data is the output dimension
def get_pred_svm(train,test):
    # Your implementation goes here
    # You may leverage the svm module from sklearn (scikit-learn)
    # return (predicted output, actual output)

    n1, m1 = train.shape  # number of rows and columns
    train_input = train.iloc[:, :m1 - 1]
    train_output = train.iloc[:, m1 - 1]

    n2, m2 = test.shape
    test_input = test.iloc[:, :m2 - 1]
    test_output = test.iloc[:, m2 - 1]

    clf = svm.SVC(C=100.0)
    #gamma Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
    clf.fit(train_input, train_output)
    pred = clf.predict(test_input)

    actual = test_output  # actual output
    pred = pd.DataFrame(pred)
    # reset indexes just in case
    actual = actual.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    output = pd.concat([pred, actual], axis=1)  # combine panda frames
    output.columns = ['pred', 'actual']  # add labels to columns
    return output

# Naive Bayes
# Assumes the last column of data is the output dimension
def get_pred_nb(train,test):
    # Your implementation goes here
    # You may leverage the naive_bayes module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    n1, m1 = train.shape  # number of rows and columns
    train_input = train.iloc[:, :m1 - 1]
    train_output = train.iloc[:, m1 - 1]

    n2, m2 = test.shape
    test_input = test.iloc[:, :m2 - 1]
    test_output = test.iloc[:, m2 - 1]

    gnb = GaussianNB() #create gaussian naive bayes object
    pred = gnb.fit(train_input, train_output).predict(test_input)

    actual = test_output  # actual output
    pred = pd.DataFrame(pred)
    # reset indexes just in case
    actual = actual.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    output = pd.concat([pred, actual], axis=1)  # combine panda frames
    output.columns = ['pred', 'actual']  # add labels to columns
    return output

# k-Nearest Neighbor
# Assumes the last column of data is the output dimension
def get_pred_knn(train,test,k):
    # Your implementation goes here
    # You may leverage the neighbors module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    n1, m1 = train.shape  # number of rows and columns
    train_input = train.iloc[:, :m1 - 1]
    train_output = train.iloc[:, m1 - 1]

    n2, m2 = test.shape
    test_input = test.iloc[:, :m2 - 1]
    test_output = test.iloc[:, m2 - 1]

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_input, train_output)
    pred = neigh.predict(test_input)

    actual = test_output  # actual output
    pred = pd.DataFrame(pred)
    # reset indexes just in case
    actual = actual.reset_index(drop=True)
    pred = pred.reset_index(drop=True)

    output = pd.concat([pred, actual], axis=1)  # combine panda frames
    output.columns = ['pred', 'actual']  # add labels to columns
    return output

def get_pred_default(train,test):
    n1, m1 = train.shape # number of rows and columns
    n2, m2 = test.shape  # number of rows and columns
    train_output = train.iloc[:, m1-1]
    train_output_counts = train_output.value_counts().sort_values()
    train_output_values = train_output_counts.keys()
    default = train_output_values[-1]
    pred = np.full(n2, default)

    pred = pd.DataFrame(pred)

    actual = test.iloc[:, m2-1]
    output = pd.concat([pred, actual], axis=1)  # combine panda frames
    output.columns = ['pred', 'actual']  # add labels to columns
    return output
##############################
# PART 2
print ('\n ############ PART 2 ############# \n')
##############################

#your implementation of do_cv_class goes here
#return pred and actual output for each test set and their corresponding fold
#Output one dataframe with n rows (n being the number of datapoints)
# First column is the prediction, second column is the true label, last column is the index of the test fold the datapoint was in.
def do_cv_class(df, num_folds, model_name):

    rows, columns = df.shape  # number of rows and columns
    partition_size = int(rows/num_folds)

    #Determine which model to use
    if model_name[-2:] == "nn":
        func = get_pred_knn
        k = model_name[:-2]
        k = int(k)
    elif model_name == "logreg":
        func = get_pred_logreg
    elif model_name == "svm":
        func = get_pred_svm
    elif model_name == "nb":
        func = get_pred_nb

    labels = ['pred', 'actual', 'folds']
    output = pd.DataFrame(columns=labels) # create panda dataframe to be outputed

    for n in range(0, num_folds):
        row_range_top = (n) * partition_size
        if n == num_folds-1: #if this is the last test set then just take the rest of the data
            row_range_top = (n) * partition_size
            test = df.iloc[row_range_top:, :]
            train = df.drop(df.index[row_range_top:])
        else:
            row_range_top = (n) * partition_size
            row_range_bottom = row_range_top + partition_size
            test = df.iloc[row_range_top:row_range_bottom, :]
            train = df.drop(df.index[row_range_top:row_range_bottom])

        if func == get_pred_knn :
            pred_actual = func(train, test,k)
        else:
            pred_actual = func(train, test) #output panda data frames with predicted and actual test ouput
        pred_actual['folds'] = n+1 #append fold
        pred_actual_fold = pred_actual
        output = output.append(pred_actual_fold,ignore_index=True) # append predict, actual test data, and fold
    return output


##############################
# PART 3
print ('\n ############ PART 3 ############# \n')

##############################

#input prediction file the first column of which is prediction value
#the 2nd column is true label (0/1)
#cutoff is a numeric value, default is 0.5
#output is a data frame with elements (tpr, fpr, acc, precision, recall)
def get_metrics(pred, cutoff=0.5):
    ### your implementation goes here

    # converting pandas dataframe to list
    # If using list comment below
    pred_data = pred.iloc[:, 0:2].copy()
    data_output = pd.np.array(pred_data)
    # If using list comment above

    predicted_output = [float(i[0]) for i in data_output] # first column of list are the Predicted values
    actual_output = [int(i[1]) for i in data_output] # Second column of list are the Actual values
    # returns list = [[FALSE, TRUE][IF CONDITION] FOR a row in LIST]
    predicted_output_manipulate = [[0, 1][x > cutoff] for x in predicted_output] # can also be performed by map(lambda x: [0, 1][x > cutoff], predicted_output)

    n11_TP = sum([[0, 1][predicted_output_manipulate[i] == 1 and actual_output[i] == 1] for i in range(len(predicted_output))])
    n00_TN = sum([[0, 1][predicted_output_manipulate[i] == 0 and actual_output[i] == 0] for i in range(len(predicted_output))])
    n10_FN = sum([[0, 1][predicted_output_manipulate[i] == 0 and actual_output[i] == 1] for i in range(len(predicted_output))])
    n01_FP = sum([[0, 1][predicted_output_manipulate[i] == 1 and actual_output[i] == 0] for i in range(len(predicted_output))])

    tpr = n11_TP/(n11_TP+n10_FN)#TP rate = TP/(TP+FN)
    fpr = n01_FP/(n00_TN+n01_FP) #FP rate = FP/(TN+FP)
    acc = (n11_TP+n00_TN)/(n11_TP+n00_TN+n10_FN+n01_FP)#Accuracy = (TP+TN)/N
    precision = n11_TP/(n11_TP+n01_FP)#precision = TP/(TP+FP)
    recall = tpr

    labels = ['tpr','fpr','acc','precision','recall']
    output = pd.DataFrame([[tpr,fpr,acc,precision,recall]],columns=labels) #create dataframe for output

    return output

####################
####import data#####
print ('\n ############ Import data ############# \n')
####################

my_data = pd.read_csv('wine.csv')
#encode class into 0/1 for easier handling by classification algorithm
my_data['type'] = np.where(my_data['type'] == 'high', 1, 0)

# scroll down to refer test cases below


##############################
# PART 4
print ('\n ############ PART 4 ############# \n')
##############################


#test cases for "do_cv_class" and "get_metrics" functions

print ('-------------------')
print ('logistic regression')
print ('-------------------')
tmp = do_cv_class(my_data,10,'logreg') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print (get_metrics(tmp.iloc[:, 0:2]))

print ('--------------------')
print ('naive Bayes')
print ('--------------------')
tmp = do_cv_class(my_data,10,'nb') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print (get_metrics(tmp.iloc[:, 0:2]))

print ('--------------------')
print ('svm')
print ('--------------------')
tmp = do_cv_class(my_data,10,'svm') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print (get_metrics(tmp.iloc[:, 0:2]))

print ('--------------------')
print ('knn')
print ('--------------------')
tmp = do_cv_class(my_data,10,'8nn') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print (get_metrics(tmp.iloc[:, 0:2]))

print ('--------------------')
print ('default')
print ('--------------------')
tmp= get_pred_default(my_data,my_data)
print_cont_table(tmp.iloc[:, 0:2])
print (get_metrics(tmp.iloc[:, 0:2]))

k_array =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
acc_array =[]
for k in range(1,21):
    knn = str(k) + "nn"
    tmp = do_cv_class(my_data, 10, knn)
    m = get_metrics(tmp.iloc[:, 0:2])
    accuracy = m.get_value(0, 'acc')
    acc_array.append(accuracy)

#scatter plot of k in "knn" and accuracy
plt.scatter(k_array, acc_array)
plt.plot(k_array, acc_array)
plt.title("Relationship between number of nearest neighbors and prediction accuracy")
plt.xlabel('k of k-nearest neighbors')
plt.ylabel('Accuracy')
plt.show()

#plot bar graph of accuracies generated by models
x = np.arange(4)
fig, ax = plt.subplots()
accuracies = [0.794712,0.839813,0.842924,0.564541]
plt.title("Bar Plot of Accuracies generated by each Model")
plt.bar(x, accuracies)
plt.xticks(x, ('Linear\n Regression', 'Naive Bayes','SVM','Default'))
plt.ylabel('Model')
plt.ylabel('Accuracy')
plt.show()

#plot bar graph of accuracies generated by all models
x = np.arange(5)
fig, ax = plt.subplots()
accuracies = [0.794712,0.839813,0.842924,0.800933,0.564541]
plt.title("Bar Plot of Accuracies generated by all Models")
plt.bar(x, accuracies)
plt.xticks(x, ('Linear\n Regression', 'Naive Bayes','SVM','8nn','Default'))
plt.ylabel('Model')
plt.ylabel('Accuracy')
plt.show()