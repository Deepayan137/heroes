import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn import pipeline, grid_search
from sklearn.model_selection import GridSearchCV
from utils.parser import params 
from utils.preproc import read_data
import pdb
import pandas as pd


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	test_1 = params["test_1"]
	test_9 = params["test_9"]
	hero_data = params["hero_data"]
	X_test, y_test, X_val, uid = read_data(test_1, test_9, hero_data, test=1)
	# X_train, y_train, X_val_t, y_val, uid = read_data(train_1, train_9, hero_data)
	
	# Model Creation
	print('processing GridSearch')
	parameters = {"max_depth": [2,3,4,5,6,7,8,9,10,11,12],"min_samples_split" :[2,3,4,5,6] ,"n_estimators" : [10, 100]    ,"min_samples_leaf": [1,2,3,4,5]    ,"max_features": (2,3,4)}
	rf_regr = RandomForestRegressor()
	ada_regr = AdaBoostRegressor()
	rf_model = GridSearchCV(rf_regr,parameters, n_jobs = -1, cv = 5, scoring=RMSE)


	# # Va;idating on the training set
	# model.fit(X_train, y_train)
	# print("Best parameters found by grid search:")
	# print(model.best_params_)
	# print("Best CV score:")
	# print(model.best_score_)
	# y_pred_val = model.predict(X_val_t)	

	# print("Mean squared error: %.2f"
 #      % mean_squared_error(y_val,y_pred_val))
	# pdb.set_trace()
	# Finally the test dataset
	rf_model.fit(X_test, y_test)
	ada_regr.fit(X_test, y_test)
	y_pred_rf = rf_model.predict(X_val)
	y_pred_ada = ada_regr.predict(X_val)
	# y_pred_linear = regr_linear.predict(X_val)
	# final_pred = (y_pred+y_pred_linear)/2
	final_pred = (y_pred_rf + y_pred_ada)/2
	
	def submission():
		data = [[uid[i], final_pred[i]] for i in range(len(final_pred))]
		df = pd.DataFrame(data, columns=["id", "kda_ratio"])
		df.to_csv("submission.csv", index=False)
	submission()


