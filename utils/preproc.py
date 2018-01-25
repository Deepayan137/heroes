import pandas as pd 
import os
import sys
from .parser import params 
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pdb
def read_data(train_1, train_9):
	df_9 = pd.read_csv(train_9)
	df_1 = pd.read_csv(train_1)
	df_1["win_ratio"] = df_1["num_wins"]/df_1["num_games"]
	df_9["win_ratio"] = df_9["num_wins"]/df_9["num_games"]
	y_train = df_9["kda_ratio"]
	y_val = df_1["kda_ratio"]
	X_train = df_9.drop(["kda_ratio"], axis=1)
	X_val = df_1.drop(["kda_ratio"], axis=1)
	return X_train, y_train, X_val, y_val
	
if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	read_data(train_1, train_9)

