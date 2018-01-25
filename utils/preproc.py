import pandas as pd 
import os
import sys
from .parser import params 
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pdb
lb_make = LabelEncoder()
def read_data(train_1, train_9, hero_data):
	df_9 = pd.read_csv(train_9)
	df_1 = pd.read_csv(train_1)
	df_hero = pd.read_csv(hero_data)
	df_1["win_ratio"] = df_1["num_wins"]/df_1["num_games"]
	df_9["win_ratio"] = df_9["num_wins"]/df_9["num_games"]
	
	
	merged_9 = df_9.merge(df_hero, on = 'hero_id', how='left')
	merged_9["primary_attr"] = lb_make.fit_transform(merged_9["primary_attr"])
	merged_9["attack_type"] = lb_make.fit_transform(merged_9["attack_type"])
	merged_9["roles"] = lb_make.fit_transform(merged_9["roles"])
	
	merged_1 = df_1.merge(df_hero, on = 'hero_id', how='left')
	merged_1["primary_attr"] = lb_make.fit_transform(merged_1["primary_attr"])
	merged_1["attack_type"] = lb_make.fit_transform(merged_1["attack_type"])
	merged_1["roles"] = lb_make.fit_transform(merged_1["roles"])
	
	

	y_train = merged_9["kda_ratio"]
	y_val = merged_1["kda_ratio"]
	X_train = merged_9.drop(["kda_ratio", "hero_id", "base_health"], axis=1)
	X_val = merged_1.drop(["kda_ratio", "hero_id", "base_health"], axis=1)
	return X_train, y_train, X_val, y_val
	
if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	hero_data = params["hero_data"]
	read_data(train_1, train_9, hero_data)

