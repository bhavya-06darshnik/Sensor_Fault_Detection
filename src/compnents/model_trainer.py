import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from xgboost import  XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    def __init__(self):

        self.model_trainer_config=ModelTrainerConfig()

        self.utils = MainUtils()

        self.models = {
                        'XGBClassifier': XGBClassifier(),
                        'GradientBoostingClassifier': GradientBoostingClassifier(),
                        'RandomForestClassifier': RandomForestClassifier(),
                        'SVC': SVC(),
        }

    def evaluate_models(self, x, y, models):
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            report={}

            for i in range(len(list[models])):
                model=list(models.values())[i]

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)

                y_test_pred = model.predict(x_test)

                train_model_score = accuracy_score(y_train, y_train_pred)

                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = train_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)



    def get_best_model(self,
                       x_train:np.array,
                       y_train:np.array,
                       x_test:np.array,
                       y_test:np.array,):

        try:


            model_report: dict=self.evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
            )

            print(model_report)

            best_model_score=max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model_object = self.models[best_model_name]
            
            return best_model_name,best_model_score,best_model_object

        except Exception as e:
            raise CustomException(e, sys)



    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            x_train,
                            y_train,) -> object:

        try:



             model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]




            grid_search: = GridSearchCV(
                 best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )

            grid_search.fit(x_train, y_train)


            best_params = grid_search.best_params_


            print("best params are:", best_params)


            finetuned_model = best_model_object.set_params(**best_params)



            return finetuned_model




        except Exception as e:
              raise CustomException(e,sys)





    def