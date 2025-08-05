## this file have code to read the data from the source and save it to the destination
## divide the data into train and test sets
## This file contains functions to handle data ingestion, including reading from various sources and saving to files
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:      # any input parameters can be added here
    train_data_path: str = os.path.join('artifacts','train.csv')    #giving input to store location of train/test/raw data
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()   #input paths are initialized/stored in the class variable

    def initiate_data_ingestion(self):      # write code here to read the code from databases 
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # reading the data from the source, here it is csv file
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # creating the directory if it does not exist, to store the train/test/raw data

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # saving the raw data to the destination path specified in the config
            logging.info('Train test Split initiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)  # splitting the data into train and test sets, 80% train and 20% test

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)  # saving the train set to the destination path specified in the config
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys) # raise custom exception to handle errors

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()