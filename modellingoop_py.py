# -*- coding: utf-8 -*-
"""modellingOOP.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q3CMFNCC2C0POq9F8Gl_XFItSBduWpVe

- Nama : Stevia Anlena Putri
- NIM : 2602092144
- Model Deployment Mid Exam

# Modelling OOP

Sekarang step selanjutnya sebelum deploy, kita buat dulu OOPnya di file py. Di sini kita kurang lebih sebenarnya hanya mengganti proses di ipynb ke suatu bentuk OOP. Kita bagi 2 kela menjadi data handler dan model handler. Data handler ini akan digunakan untuk load data dan menentukan target kolomnya, sedangkan model handler ini untuk split data, handling data, encode, dan fit datanya ke model RF dan XGBoost kita. NA values akan dihandle dengan cara panggil function check outliernya pada kelas model handler dan hitung hasil outliernya. Apabila ada outlier (>0) maka akan diisi dengan median, tetapi jika tidak ada, akan diisi dengan mean. Encoding process untuk Gender dan Geography akan disave ke pickle masing-masing. Setelah selesai diproses datanya, barulah kita panggil function RF dan XGB untuk fit datanya dan check accuracynya. Di sini kita bisa lihat meskipun sedikit berbeda dengan yang di ipynb akurasinya, tetapi hanya selisih 1 atau 2 nilai saja dan untuk bagian recall dan precision masih lebih seimbang XGBoost. Maka, kita save XGBoostnya menjadi best model pickle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from xgboost import XGBClassifier


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def create_input_output(self, target_column, drop1, drop2, drop3, drop4):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(columns=[target_column, drop1, drop2, drop3, drop4])

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model = None

    def split_data(self, test_size=0.3, random_state=0):
        self.x_train, self.x_temp, self.y_train, self.y_temp = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
            self.x_temp, self.y_temp, test_size=0.5, random_state=random_state)

    def checkCreditScoreOutlierWithBox(self, kolom):
        boxplot = self.x_train.boxplot(column=[kolom])
        plt.show()

    def createMedianFromColumn(self, kolom):
        return self.x_train[kolom].median()

    def createMeanFromColumn(self, kolom):
        return self.x_train[kolom].mean()

    def fillingNAWithNumbers(self, columns, number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_val[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)

    def handle_missing_value(self, column):
        #Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = self.x_train[column].quantile(0.25)
        Q3 = self.x_train[column].quantile(0.75)

        #Calculate the interquartile range (IQR)
        IQR = Q3 - Q1

        #Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Check for outliers
        outliers = self.x_train[(self.x_train[column] < lower_bound) | (self.x_train[column] > upper_bound)]
        return len(outliers)


    def gender_encode(self):
        gender_encode = {"Gender": {"Male": 1, "Female": 0}}
        self.x_train = self.x_train.replace(gender_encode)
        self.x_val = self.x_val.replace(gender_encode)
        self.x_test = self.x_test.replace(gender_encode)
        filename = 'gender_encode.pkl'
        pickle.dump(gender_encode, open(filename, 'wb'))

    def label_encode_geo(self):
        geo_encode = {"Geography": {"France": 2, "Spain": 1, "Germany": 0, "Others": 3}}
        self.x_train = self.x_train.replace(geo_encode)
        self.x_val = self.x_val.replace(geo_encode)
        self.x_test = self.x_test.replace(geo_encode)
        filename = 'label_encode_geo.pkl'
        pickle.dump(geo_encode, open(filename, 'wb'))

    def makePrediction(self):
        if self.model is not None:
            self.y_val_pred = self.model.predict(self.x_val)
            self.y_test_pred = self.model.predict(self.x_test)
        else:
            print("Model has not been trained yet.")

    def createReport(self):
        print('\nClassification Report for Validation Data:\n')
        print(classification_report(self.y_val, self.y_val_pred))
        print('\nClassification Report for Test Data:\n')
        print(classification_report(self.y_test, self.y_test_pred))

    def train_model(self, model):
        model.fit(self.x_train, self.y_train)
        self.model = model

    def tuningParameterRF(self):
        parameters = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5]
        }
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(rf,
                                  param_grid=parameters,
                                  scoring='accuracy')
        grid_search.fit(self.x_train, self.y_train)

        print("Grid Search Best Score:", grid_search.best_score_)
        print("Best Params:", grid_search.best_params_)

        best_rf_model = grid_search.best_estimator_
        self.train_model(best_rf_model)
        self.makePrediction()
        self.createReport()

    def tuningParameterXGB(self):
        parameters = {
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 4, 5],
            'n_estimators': [100, 200, 300]
        }
        xgb = XGBClassifier()
        grid_search = GridSearchCV(xgb ,
                            param_grid=parameters,
                            scoring='accuracy')
        grid_search.fit(self.x_train, self.y_train)
        print("Best params:", grid_search.best_params_)

        best_xgb_model = grid_search.best_estimator_
        self.train_model(best_xgb_model)
        self.makePrediction()
        self.createReport()

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

file_path = 'data_D.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn', 'Unnamed: 0', 'id', 'CustomerId', 'Surname')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

# Check data shapes
print(model_handler.x_train.shape)
print(model_handler.x_val.shape)
print(model_handler.x_test.shape)
print(model_handler.y_train.shape)
print(model_handler.y_val.shape)
print(model_handler.y_test.shape)

# Apply preprocessing steps
model_handler.checkCreditScoreOutlierWithBox('CreditScore')
outlier = model_handler.handle_missing_value('CreditScore')
if outlier > 0:
  cs_replace_na = model_handler.createMedianFromColumn('CreditScore')
  model_handler.fillingNAWithNumbers('CreditScore', cs_replace_na)
else:
  cs_replace_na = model_handler.createMeanFromColumn('CreditScore')
  model_handler.fillingNAWithNumbers('CreditScore', cs_replace_na)

model_handler.gender_encode()
model_handler.label_encode_geo()

# Model 1 : RF
model_handler.tuningParameterRF()

# Model 2 : XGB
model_handler.tuningParameterXGB()

# Save the best model to a file
model_handler.save_model_to_file('best_model.pkl')