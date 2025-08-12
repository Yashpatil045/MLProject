## data related to transformation processes, such as normalization, encoding, or feature extraction.
## eg. label encoding, one-hot encoding, scaling, etc.
## This file contains functions to transform the data for better model performance.
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ["age", "salary"]
            categorical_features = ["gender", "city"]

            numerical_pipeline = pipeline.Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = pipeline.Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = pipeline.ColumnTransformer(transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException("Error occurred while creating data transformer object", sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Data loaded successfully")

            preprocessor = self.get_data_transformer_object()

            X_train = train_data.drop("target", axis=1)
            y_train = train_data["target"]

            X_test = test_data.drop("target", axis=1)
            y_test = test_data["target"]

            logging.info("Data split into train and test sets")

            # Fit the preprocessor on the training data
            preprocessor.fit(X_train)

            # Transform the data
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Data transformation complete")

            return X_train_transformed, y_train, X_test_transformed, y_test

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException("Error occurred during data transformation", sys) from e  