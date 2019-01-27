import pandas as pd
import numpy as np
import os.path
try: 
	import urllib.request as urllib
except:
	import urllib as urllib

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

#### Use this template file to solve the problem

###### DONOT CHANGE THIS ############
### First we download data files from github

url1 = "https://raw.githubusercontent.com/Auquan/qq-ic/master/training_data.csv"
url2 = "https://raw.githubusercontent.com/Auquan/qq-ic/master/test_data.csv"
if not (os.path.isfile("training_data.csv")):
	urllib.urlretrieve(url1, "training_data.csv")
if not (os.path.isfile("test_data.csv")):
	urllib.urlretrieve(url2, "test_data.csv")


### This command reads CSV into a dataframe
market_data = pd.read_csv("training_data.csv", index_col='TimeStamp')
print(market_data)
training_X = market_data.drop(columns=['Y'])
training_y = market_data['Y']

#### Now you can play with data

#### Variable Selection

#### Assess Correlation 

#### Assess change in correlation with time

#### Multi Linear regression Fitting the model
model = linear_model.LinearRegression() ### You can use any model of your choice
model.fit(training_X, training_y) #Create the linear regression


#### Compare Models


#### Cross Validation - Remember this is time series data!! Don't use information from the future

#### Check Relative Importance

#### Try GLM/Nonlinear Regression



#### Make predictions for new data using model of your choice
## This assumes that you have fitted a model named 'model'

test_X = pd.read_csv("test_data.csv", index_col='TimeStamp')
z = model.predict(test_X)

### DO NOT CHANGE THIS
test_y = pd.DataFrame(z, index=test_X.index, columns=['fit'])
print(test_y)
##### Save your prediction as CSV

test_y.to_csv('predictions.csv', float_format='%.3f', header=True, index=True, index_label='TimeStamp')
