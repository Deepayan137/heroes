import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.parser import params 
from utils.preproc import read_data
import pdb
import pandas as pd

if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	test_1 = params["test_1"]
	test_9 = params["test_9"]
	hero_data = params["hero_data"]
	X_train, y_train, X_val, = read_data(test_1, test_9, hero_data, test=1)
	
	regr_linear = linear_model.LinearRegression()
	regr = RandomForestRegressor(n_estimators=100, max_depth=19)
	regr.fit(X_train, y_train)
	regr_linear.fit(X_train, y_train)
	y_pred = regr.predict(X_val)
	y_pred_linear = regr_linear.predict(X_val)
	final_pred = (y_pred+y_pred_linear)/2
	
	def submission():
		data = [[X_val["id"].iloc[i], final_pred[i]] for i in range(len(final_pred))]
		df = pd.DataFrame(data, columns=["id", "kda_ratio"])
		df.to_csv("submission.csv", index=False)
	submission()

	# print("Mean squared error: %.2f"
 #      % mean_squared_error(y_val,final_pred))
