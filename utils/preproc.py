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


def read_data(train_1, train_9, hero_data, test=None):
	df_9 = pd.read_csv(train_9)
	df_1 = pd.read_csv(train_1)
	df_hero = pd.read_csv(hero_data)
	uid = df_1["id"]
	def transform(df, **kwargs):
		attributes = kwargs["attributes"]
		for attribute in attributes:
			df[attribute] = lb_make.fit_transform(df[attribute])
		return df

	if test != None:

		df_9 = pd.read_csv(train_9)
		df_1 = pd.read_csv(train_1)
		uid = df_1["id"]
		merged_9 = df_9.merge(df_hero, on = 'hero_id', how='left')
		merged_1 = df_1.merge(df_hero, on = 'hero_id', how='left')
		merged_9 = transform(merged_9, attributes=["primary_attr", "attack_type","roles"])
		merged_1 = transform(merged_1, attributes=["primary_attr", "attack_type","roles"])
		y_train = merged_9["kda_ratio"]
		X_train = merged_9.drop(["kda_ratio", "base_mana", "user_id", "id",  "hero_id", "base_health"], axis=1)
		X_test = merged_1.drop(["hero_id",  "user_id", "id", "base_mana", "base_health"], axis=1)
		return X_train, y_train, X_test, uid

	merged_9 = df_9.merge(df_hero, on = 'hero_id', how='left')
	merged_1 = df_1.merge(df_hero, on = 'hero_id', how='left')
	merged_9 = transform(merged_9, attributes=["primary_attr", "attack_type","roles"])
	merged_1 = transform(merged_1, attributes=["primary_attr", "attack_type","roles"])
	# merged_1["win_ratio"] = merged_1["num_wins"]/merged_1["num_games"]
	# merged_9["win_ratio"] = merged_9["num_wins"]/merged_9["num_games"]
	y_train = merged_9["kda_ratio"]
	y_val = merged_1["kda_ratio"]
	X_train = merged_9.drop(["kda_ratio", "user_id", "id", "hero_id", "base_mana", "base_health"], axis=1)
	X_val = merged_1.drop(["kda_ratio",  "user_id", "id", "hero_id", "base_mana", "base_health"], axis=1) 
	return X_train, y_train, X_val, y_val, uid
	
if __name__ == '__main__':
	train_1 = params["train_1"]
	train_9 = params["train_9"]
	hero_data = params["hero_data"]
	read_data(train_1, train_9, hero_data)

