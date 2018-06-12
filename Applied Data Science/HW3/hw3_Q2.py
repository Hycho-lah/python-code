#######################################################
# Homework 3
# Name: Elissa Ye
# andrew ID: eye
# email: eye@andrew.cmu.edu
#######################################################

# Problem 2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn import linear_model

df = pd.read_csv('BMI.csv')
n, m = df.shape  # number of rows and columns

#2b Correlation based feature selection
#takes a data frame as input (assume that the last column is the output dimension).
# This function will screen the available input features in a loop and compute
# the absolute values of linear correlation coefficients between each input and the output.
# The function will output a sorted data frame containing the names of features and the corresponding coefficients.
def filter_features_by_cor(df):
    output = df.iloc[:,m-1] #output would always be the last column fat percentage
    output_list = output.tolist()
    corrcoef_array = []
    for i in range(0,m-2):
        input_list = df.iloc[:,i].tolist()
        pair_v = [input_list, output_list]
        corrcoef = abs(np.corrcoef(pair_v)) #each row should represent a variable
        corrcoef_array = np.append(corrcoef_array,corrcoef[0,1])
    feature_names = list(df)
    feature_names = feature_names[0:m-2]
    output_df = pd.DataFrame(feature_names, columns=['Features'])
    output_df['CorrCoef'] = corrcoef_array
    output_df = output_df.sort_values('CorrCoef')
    output_df = output_df.reset_index()
    output_df = output_df.drop(columns = "index")
    return output_df

output_df = filter_features_by_cor(df)
print(output_df)

# 2c - Univariate feature selection works by selecting the best features based on univariate statistical tests.
# It can be seen as a preprocessing step to an estimator.

train_input = df.iloc[:,:m-1]
train_output = df.iloc[:,m-1]

#test out top features for size k using univariate feature selection
df_set = SelectKBest(f_regression, k=1).fit_transform(train_input, train_output)
print(df_set)
df_set = SelectKBest(f_regression, k=2).fit_transform(train_input, train_output)
print(df_set)
df_set = SelectKBest(f_regression, k=3).fit_transform(train_input, train_output)
print(df_set)

#2d - Backward stepwise regression
estimator = linear_model.LinearRegression()
selector = RFECV(estimator, cv=10, step=1)
selector = selector.fit(train_input, train_output)
BMI_features = list(df)
BMI_features = BMI_features[0:m-1]
print(BMI_features )
print(selector.support_)
print(selector.ranking_)


