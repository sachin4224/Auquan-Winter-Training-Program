import pandas as pd
import numpy as np
import os.path
try: 
	import urllib.request as urllib
except:
	import urllib as urllib

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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
#market_data = pd.read_csv("training_data.csv", index_col='TimeStamp')
#market_data = pd.read_csv("training_data.csv")
#print(market_data)
#columns = market_data.columns.values[1:-1]
#training_X = market_data.drop(columns=['Y'])
#training_X = market_data.iloc[:,0:-1]
#training_y = market_data['Y']

#### Now you can play with data

path = "./training_data.csv"
path1 = "./test_data.csv"
market_data = pd.read_csv(path)
columns = market_data.columns.values
    
columns = columns[1:-1]
training_X = market_data.iloc[:,1:-1]

indexs = market_data.index
training_y=market_data['Y']


X = training_X.values
Y= training_y.values


lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(Y)

model = ExtraTreesClassifier()
model.fit(X, encoded)


#### Variable Selection
sortlist = np.argsort(model.feature_importances_)


variables=sortlist[-25:]


regressors = pd.DataFrame()
for column in variables:
	temp = pd.DataFrame({ '%d' % column:X[:,column]})
	regressors = pd.concat( [regressors , temp], axis = 1 )



#### Assess Correlation 

corr_matrix = regressors.corr()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]


regressors = regressors.drop(to_drop,axis=1)


#### Assess change in correlation with time




#### Multi Linear regression Fitting the model
scaler = StandardScaler()
norm_regressors = scaler.fit_transform( regressors )


# model = linear_model.LinearRegression() ### You can use any model of your choice
model = RandomForestRegressor(n_estimators=100,random_state=10)
model.fit(norm_regressors,training_y)


#### Compare Models
# train_features, test_features, train_labels, test_labels = train_test_split(norm_regressors, training_y, test_size = 0.25, random_state = 42)


#### Cross Validation - Remember this is time series data!! Don't use information from the future
model1 = RandomForestRegressor(n_estimators=50,random_state=10)

kf = KFold(n_splits=2)

KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
	# print("TRAIN:", train_index, "TEST:", test_index)
	train_features, test_features = norm_regressors[train_index], norm_regressors[test_index]
	train_labels,test_labels = training_y[train_index], training_y[test_index]


model1.fit(train_features,train_labels)
predictions = model.predict(test_features)
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Accuracty:', 100 - round(np.mean(errors), 2), 'degrees.')

#### Check Relative Importance

#### Try GLM/Nonlinear Regression

# model1 = RandomForestRegressor(n_estimators=100,random_state=10)
# model1.fit(norm_regressors,training_y)


#### Make predictions for new data using model of your choice
## This assumes that you have fitted a model named 'model'

test= pd.read_csv(path1)
test1 = test.iloc[:,1:]


test1 = test1.values

test_X = pd.DataFrame()
for column in variables:
	temp = pd.DataFrame({ '%d' % column:test1[:,column]})
	test_X = pd.concat( [test_X , temp], axis = 1 )

test_X = test_X.drop(to_drop,axis=1)
norm_test = scaler.transform( test_X )

z = model.predict(norm_test)


### DO NOT CHANGE THIS
test_y = pd.DataFrame(z, index=test_X.index, columns=['fit'])

##### Save your prediction as CSV

test_y.to_csv('predictions.csv', float_format='%.3f', header=True, index=True, index_label='TimeStamp')
