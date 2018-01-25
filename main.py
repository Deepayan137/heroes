import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from utils.parser import params 
from utils.preproc import read_data
import pdb
import pandas as pd

if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	hero_data = params["hero_data"]
	X_train, y_train, X_val, y_val = read_data(train_1, train_9, hero_data)
	# pdb.set_trace()
	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_val)
	def submission():
		data = [[X_val["id"].iloc[i], y_pred[i]] for i in range(len(y_pred))]
		df = pd.DataFrame(data, columns=["id", "kda_ratio"])
		df.to_csv("submission.csv", index=False)
	submission()
	
	print("Mean squared error: %.2f"
      % mean_squared_error(y_val,y_pred))
