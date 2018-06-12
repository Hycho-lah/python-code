# coding: utf-8

#######################################################
# Homework 3
# Name: Elissa Ye
# andrew ID: eye
# email: eye@andrew.cmu.edu
#######################################################

# Problem 1

##############################
# prepare data for PCA analysis
##############################
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('SP500_close_price.csv')
df['date'] = pd.to_datetime(df['date'])  # convert date to datetime format
df.index = df['date']  # sets date as index
del df['date']  # deletes a column
df = df.fillna(method='ffill')  # fill na with last valid value
dflog = df.apply(np.log)
# compute absolute returns on the log transformed data
daily_return = dflog.diff(periods=1).dropna()

daily_return_std = preprocessing.StandardScaler().fit_transform(daily_return.T) #normalize data
daily_return_std = daily_return_std.T #transpose data again to original data
pca = PCA()  # create PCA object
pca_data = pca.fit_transform(daily_return_std) #rows are PC

#1a
#define scree plot axes
var = np.round(pca.explained_variance_, decimals=2)  #variance
components = []
for l in range(1, len(var)+1):
    components = np.append(components,l)

#plot scree plot
plt.scatter(components,var)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

#cumulative percentage of variance retained
var_ratio = np.round(pca.explained_variance_ratio_, decimals=10) #take percent variance
var_ratio_cum = np.cumsum(var_ratio) #get cumulative ratio sums

#plot cumulative percentage of variance
plt.scatter(components,var_ratio_cum)
plt.ylabel('Cumulative percentage of variance retained')
plt.xlabel('Number of Principal Components Kept')
plt.title('Number of Components kept vs. Cumulative variance retained')
plt.show()

#find # of principle components to retain
for x in range(0,len(var_ratio_cum)):#obtain components to be kept in order to retain 80% of total variance
    if var_ratio_cum[x] >= 0.8:
        print(components[x])
        break

#use these values to get reconstruction error
print(var_ratio_cum[1])#obtain percent variance retained for 2 PCA
print(sum(var))#obtain total variance

#1b
#plot time series vs. PC1
dates = daily_return.index.get_values()
plt.scatter(dates,pca_data[:,0]) #first column = first component
plt.ylabel('PC1 values')
plt.xlabel('Dates')
plt.title('Times series vs. PC1')
plt.show()

#extract weights
print(pca.components_[0])
print(pca.components_[1])

#load tickers
tickers = pd.read_csv('SP500_ticker.csv',encoding = "ISO-8859-1")

#create for PC1 plot
stocks = list(daily_return)
d = {'stocks': stocks, 'weights': pca.components_[0]}
stocks_weights = pd.DataFrame(data=d)

industrials = []
financials = []
health_care = []
consumer_staples =[]
information_technology =[]
utilities = []
consumer_discretionary = []
telecommunications_services =[]
energy = []
materials =[]

#sort sectors
for i in range(0,len(stocks_weights)):
    stock_name = stocks_weights.iloc[i, 0]
    ticker = tickers[tickers['ticker'].str.match(stock_name)] #get the ticker that has the stock name
    if ticker.iloc[0,2] == "Industrials ":
        industrials = np.append(industrials,stocks_weights.iloc[i,1])
    elif ticker.iloc[0,2] == "Financials ":
        financials = np.append(financials, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0,2] == "Health Care ":
        health_care = np.append(health_care, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Consumer Staples ":
        consumer_staples = np.append(consumer_staples, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Information Technology ":
        information_technology = np.append(information_technology, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Utilities ":
        utilities = np.append(utilities, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Consumer Discretionary ":
        consumer_discretionary = np.append(consumer_discretionary, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Telecommunications Services ":
        telecommunications_services = np.append(telecommunications_services, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Energy ":
        energy = np.append(energy, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Materials ":
        materials = np.append(materials, stocks_weights.iloc[i, 1])

mean_weights = [np.mean(industrials),np.mean(financials),np.mean(health_care),np.mean(consumer_staples),np.mean(information_technology), np.mean(utilities),np.mean(consumer_discretionary),np.mean(telecommunications_services),np.mean(energy),np.mean(materials)]

#plot PC1 plot
x = np.arange(10)
tickers_array = ["Industrials", "Financials", "Healthcare", "Consumer Staples","Information Technology","Utilities","Consumer Discretionary", "Telecommunications Services","Energy","Materials"]
fig, ax = plt.subplots()
plt.bar(x, mean_weights)
plt.xticks(x, tickers_array,rotation=60)
plt.ylabel("Industry Sector")
plt.xlabel("Weights")
plt.title("Industry Sector vs. PC1 Weights")
plt.show()

#repeat for PC2
stocks = list(daily_return)
d = {'stocks': stocks, 'weights': pca.components_[1]}
stocks_weights = pd.DataFrame(data=d)

industrials = []
financials = []
health_care = []
consumer_staples =[]
information_technology =[]
utilities = []
consumer_discretionary = []
telecommunications_services =[]
energy = []
materials =[]

#sort sectors
for i in range(0,len(stocks)):
    stock_name = stocks_weights.iloc[i, 0]
    ticker = tickers[tickers['ticker'].str.match(stock_name)] #get the ticker that has the stock name
    if ticker.iloc[0,2] == "Industrials ":
        industrials = np.append(industrials,stocks_weights.iloc[i,1])
    elif ticker.iloc[0,2] == "Financials ":
        financials = np.append(financials, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0,2] == "Health Care ":
        health_care = np.append(health_care, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Consumer Staples ":
        consumer_staples = np.append(consumer_staples, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Information Technology ":
        information_technology = np.append(information_technology, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Utilities ":
        utilities = np.append(utilities, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Consumer Discretionary ":
        consumer_discretionary = np.append(consumer_discretionary, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Telecommunications Services ":
        telecommunications_services = np.append(telecommunications_services, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Energy ":
        energy = np.append(energy, stocks_weights.iloc[i, 1])
    elif ticker.iloc[0, 2] == "Materials ":
        materials = np.append(materials, stocks_weights.iloc[i, 1])

mean_weights = [np.mean(industrials),np.mean(financials),np.mean(health_care),np.mean(consumer_staples),np.mean(information_technology), np.mean(utilities),np.mean(consumer_discretionary),np.mean(telecommunications_services),np.mean(energy),np.mean(materials)]

#plot PC2 plot
x = np.arange(10)
tickers_array = ["Industrials", "Financials", "Healthcare", "Consumer Staples","Information Technology","Utilities","Consumer Discretionary", "Telecommunications Services","Energy","Materials"]
fig, ax = plt.subplots()
plt.bar(x, mean_weights)
plt.xticks(x, tickers_array,rotation=60)
plt.ylabel("Industry Sector")
plt.xlabel("Weights")
plt.title("Industry Sector vs. PC2 Weights")
plt.show()
