## Handles transformation: encoding, scaling, etc.

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, target_column):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = target_column

    def get_data_transformer_object(self, X):
        try:
            # Detect numerical & categorical features automatically
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            logging.info(f"Numerical Features: {numerical_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error creating data transformer object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f"Loaded train & test datasets")

            if self.target_column not in train_data.columns:
                raise CustomException(f"Target column '{self.target_column}' not found in dataset", sys)

            X_train = train_data.drop(self.target_column, axis=1)
            y_train = train_data[self.target_column]
            X_test = test_data.drop(self.target_column, axis=1)
            y_test = test_data[self.target_column]

            preprocessor = self.get_data_transformer_object(X_train)

            # Fit and transform
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation completed successfully")
            return X_train_transformed, y_train, X_test_transformed, y_test

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)