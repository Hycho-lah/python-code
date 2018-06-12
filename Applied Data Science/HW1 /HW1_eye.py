import numpy as np
import scipy.stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#defining brief function
def brief(df):
    num_rows, num_columns = df.shape

    print("\nThis data set has", num_rows,"Rows", num_columns,"Attributes")

    #String names
    attribute_ID = "Attribute_ID"
    attribute_name = "Attribute_Name"
    missing = "Missing"
    mean = "Mean"
    median = "Median"
    sdev = "Sdev"
    min = "Min"
    max = "Max"
    arity = "Arity"
    mcvs_counts = "MCVs_counts"

    print("\nreal valued attributes")
    print("----------------------")
    print(attribute_ID.rjust(14), attribute_name.rjust(25),missing.rjust(8),mean.rjust(10),median.rjust(10),sdev.rjust(10),min.rjust(10),max.rjust(10))

    attribute_names = list(df)
    attribute_ID_val = 0
    missing_val = 0
    mean_val = 0.0
    median_val = 0
    sdev_val = 0.0
    min_val = 0
    max_val = 0
    arity_val = 0
    mcvs_counts_val = ""


    r = 0 #value of row
    for i in range(0, num_columns):
        #first determine whether attribute is real
        if pd.api.types.is_numeric_dtype(df[attribute_names[i]]):
            attribute_values = df.iloc[:, i]
            attribute_ID_val = i+1
            attribute_name_val = attribute_names[i]

            missing_val = attribute_values.isnull().sum() #obtain missing values

            #calculate mean
            mean_val_raw = np.nanmean(attribute_values)
            mean_val = round(mean_val_raw, 2) # round to two decimal points
            mean_val = '%.2f' % mean_val #preserve two decimal points

            #calculate median
            median_val_raw = np.nanmedian(attribute_values)
            median_val = round(median_val_raw, 2)
            median_val = '%.2f' % median_val

            #calculate standard deviation
            sdev_val_raw = np.nanstd(attribute_values)
            sdev_val = round(sdev_val_raw, 2)
            sdev_val = '%.2f' % sdev_val

            #obtain min
            min_val = np.amin(attribute_values)
            min_val = '%.2f' % min_val

            #obtain max
            max_val = np.amax(attribute_values)
            max_val = '%.2f' % max_val

            r += 1  # value of row
            print (r,repr(attribute_ID_val).rjust(12), attribute_name_val.rjust(25),repr(missing_val).rjust(8),mean_val.rjust(10),median_val.rjust(10),sdev_val.rjust(10),min_val.rjust(10),max_val.rjust(10))


    print("symbolic attributes")
    print("-------------------")
    print(attribute_ID.rjust(14), attribute_name.rjust(25),missing.rjust(8),arity.rjust(10),mcvs_counts.rjust(21))

    r = 0 #value of row
    for i in range(0, num_columns):
        #is_string_dtype
        if pd.api.types.is_string_dtype(df[attribute_names[i]]):
            attribute_values = df.iloc[:, i]
            attribute_ID_val = i + 1
            attribute_name_val = attribute_names[i]

            missing_val = attribute_values.isnull().sum()

            arity_val = attribute_values.nunique() #counts number of unique symbols

            #count MCV value frequencies and sort
            MCV = attribute_values.value_counts().sort_values()
            MCV_names = MCV.keys()
            MCV_1_name = MCV_names[-1]
            MCV_1_value = MCV.iloc[-1]
            mcvs_counts_val = str(MCV_1_name) + "(" + str(MCV_1_value) + ")"
            if len(MCV) > 1:
                MCV_2_name = MCV_names[-2]
                MCV_2_value = MCV.iloc[-2]
                mcvs_counts_val += " " + str(MCV_2_name) + "(" + str(MCV_2_value) + ")"
                if len(MCV) > 2:
                    MCV_3_name = MCV_names[-3]
                    MCV_3_value = MCV.iloc[-3]
                    mcvs_counts_val += " " + str(MCV_3_name) + "(" + str(MCV_3_value) + ")"

                #counts = attribute_values.count("No") # count the number of occurences of particular symbol
                #MSV[s] = counts
            r += 1  # value of row
            print(r, repr(attribute_ID_val).rjust(12), attribute_name_val.rjust(25), repr(missing_val).rjust(8), repr(arity_val).rjust(10),mcvs_counts_val.rjust(21))

#Q2 Code
# Connect-the-dots model that learns from train set and is being tested using test set
# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension
def get_pred_dots(train,test):
    n,m = train.shape # number of rows and columns
    X = train.iloc[:,:m-1]# get training input data
    query = test.iloc[:,:m-1]# get test input data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)
    distances, nn_index = nbrs.kneighbors(query)# Get two nearest neighbors
    pred = (train.iloc[nn_index[:,0],m-1].values+train.iloc[nn_index[:,1],m-1].values)/2.0
    return pred

# Linear model
# Assumes the last column of data is the output dimension
def get_pred_lr(train,test):
    n, m = train.shape  # number of rows and columns
    train_input = train.iloc[:,:m-1]
    train_output = train.iloc[:,m-1]
    test_input = test.iloc[:,:m-1]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(train_input, train_output)
    pred = regr.predict(test_input)
    return pred

# Default predictor model
# Assumes the last column of data is the output dimension
def get_pred_default(train,test):
    n, m = test.shape # number of rows and columns
    train_output = train.iloc[:, m-1]
    train_output_counts = train_output.value_counts().sort_values()
    train_output_values = train_output_counts.keys()
    default = train_output_values[-1]
    pred = np.full(n, default)
    return pred

def do_cv(df,output,k,func):
    #reorganize data frame so that last column is the output variable
    output_variable = df.loc[:,output]
    df_dropped = df.drop([output], axis=1)
    df_dropped[output] = output_variable

    num_rows, num_columns = df_dropped.shape  # number of rows and columns
    partition_size = int(num_rows/k)

    #shuffle data frame
    df_random = df_dropped.sample(frac=1)

    MSE_array = []
    for n in range(0, k):
        row_range_top = (n)*partition_size
        row_range_bottom = row_range_top + partition_size
        test = df_random.iloc[row_range_top:row_range_bottom,:]
        train = df_random.drop(df_random.index[row_range_top:row_range_bottom])
        pred = func(train,test)
        true_output = test.iloc[:,-1]
        true_output_array = true_output.as_matrix()
        MSE = mean_squared_error(true_output_array, pred)
        MSE_array.append(MSE)
    return MSE_array

if __name__ == "__main__":
    # Assign spreadsheet filename to `file`
    file = 'house_no_missing.csv'

    # Load data in dataframe
    df = pd.read_csv(file)

    # print heading for Q1
    num_tilde = 26 + len(file)
    for i in range(0, num_tilde):
        print("~", end="")
    print("\nbrief function output for ", end="")
    print(file)
    for i in range(0, num_tilde):
        print("~", end="")
    brief(df)

    #histogram of house_values
    house_values = df.loc[:, 'house_value']
    plt.hist(house_values, bins=15)
    plt.ylabel('Frequency')
    plt.xlabel('House Values')
    plt.title("Distribution of House Values")
    plt.show()

    #scatter plots
    crime_rate = df.loc[:, 'Crime_Rate']
    plt.scatter(crime_rate, house_values)
    plt.title("Relationship between crime rate and house value")
    plt.xlabel('Crime Rate')
    plt.ylabel('House Values')
    plt.show()
    dist_to_employment_center = df.loc[:, 'dist_to_employment_center']
    plt.scatter(dist_to_employment_center, house_values)
    plt.title("Relationship between distance to employment center and house value")
    plt.xlabel('Distance to employment center')
    plt.ylabel('House Values')
    plt.show()
    property_tax = df.loc[:, 'property_tax_rate']
    plt.scatter(property_tax, house_values)
    plt.title("Relationship between property tax and house value")
    plt.xlabel('Property Tax')
    plt.ylabel('House Values')
    plt.show()
    nitric_oxides = df.loc[:, 'Nitric_Oxides']
    plt.scatter(nitric_oxides, house_values)
    plt.title("Relationship between Nitric Oxides and house value")
    plt.xlabel('Nitric Oxides')
    plt.ylabel('House Values')
    plt.show()
    num_of_rooms = df.loc[:, 'num_of_rooms']
    plt.scatter(num_of_rooms, house_values)
    plt.title("Relationship between number of rooms and house value")
    plt.xlabel('Number of Rooms')
    plt.ylabel('House Values')
    plt.show()
    student_teacher_ratio = df.loc[:, 'student_teacher_ratio']
    plt.scatter(student_teacher_ratio, house_values)
    plt.title("Relationship between student-teacher ratio and house value")
    plt.xlabel('Student teacher ratio')
    plt.ylabel('House Values')
    plt.show()
    accessiblity_to_highway = df.loc[:, 'accessiblity_to_highway']
    plt.scatter(accessiblity_to_highway, house_values)
    plt.title("Relationship between accessibility to highway and house value")
    plt.xlabel('Accessibility to Highway')
    plt.ylabel('House Values')
    plt.show()

    #bar plot for charles river bound house value
    charles_river_bound_yes = df.loc[df['Charles_river_bound'] == "Yes"]
    charles_river_bound_no = df.loc[df['Charles_river_bound'] == "No"]
    charles_river_bound_yes = charles_river_bound_yes.loc[:,'house_value']
    charles_river_bound_no = charles_river_bound_no.loc[:, 'house_value']
    charles_river_bound_yes_mean = pd.DataFrame.mean(charles_river_bound_yes)
    charles_river_bound_no_mean = pd.DataFrame.mean(charles_river_bound_no)
    charles_river_bound_means = [charles_river_bound_yes_mean,charles_river_bound_no_mean]
    m = np.arange(2)
    fig, ax = plt.subplots()
    plt.title("House Value based on whether Charles River Bound")
    plt.bar(m, charles_river_bound_means)
    plt.xticks(m, ('River Bound', 'Not River Bound'))
    plt.ylabel('House Value')
    plt.show()

    # slice df columns for Q2
    input_crime_rate = df.loc[:, 'Crime_Rate']
    input_crime_rate_log = input_crime_rate.apply(np.log)
    output_house_value = df.loc[:, 'house_value']

    # initialize and create a dataframe "df_q" with attributes log(Crime_Rate) and house_value
    df_q = pd.DataFrame()
    df_q['log_crime_rate'] = input_crime_rate
    df_q['house_value'] = output_house_value

    # Q2bi Apply leave-one-out cross-validation for three models
    num_rows, num_columns = df_q.shape
    lou_dots = do_cv(df_q, "house_value", num_rows, get_pred_dots)
    lou_lr = do_cv(df_q, "house_value", num_rows, get_pred_lr)
    lou_default = do_cv(df_q, "house_value", num_rows, get_pred_default)

    # 2bii
    # compute standard deviation
    std_dots = np.std(lou_dots)
    std_lr = np.std(lou_lr)
    std_default = np.std(lou_default)
    # computer standard error = std/sqrt(n)
    str_dots = std_dots / np.sqrt(len(lou_dots))
    str_lr = std_lr / np.sqrt(len(lou_lr))
    str_default = std_default / np.sqrt(len(lou_default))

    # 95% confidence interval = average +- 1.96*SE
    str_array = [str_dots * 1.96, str_lr * 1.96, str_default * 1.96]

    #calculate average MSE for each model
    MSE_avg_dots = np.mean(lou_dots)
    MSE_avg_lr = np.mean(lou_lr)
    MSE_avg_default = np.mean(lou_default)

    MSE_avgs = [MSE_avg_dots, MSE_avg_lr, MSE_avg_default]

    #Plot bar plot for leave-one-out cross validation for three models
    x = np.arange(3)
    fig, ax = plt.subplots()
    plt.title("Bar Plot of MSEs generated by each Model\nusing log(Crime Rate) to estimate House Value")
    plt.bar(x, MSE_avgs, yerr=str_array)
    plt.xticks(x, ('Connect-the-dots', 'Linear Regression', 'Default'))
    plt.ylabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.show()
