import os 
import sys 
from dataclasses import dataclass 
import pickle

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import StandardScaler, OrdinalEncoder # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from src.exception import CustomException 
from src.logger import logging 
from src.utils import save_function
from src.utils import Project_dir_path


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(Project_dir_path(), 'artifacts\\preprocessor.pkl')

class DataTransformation: 
    def __init__(self): 
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self): 
        try: 
            logging.info("Data Transformation has been initiated") 
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
            numerical_cols = ['Year', 'Selling_Price', 'Kms_Driven', 'Owner']
            
            # Define the custom ranking for each ordinal variable
            car_categories = ['ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
            's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
            '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
            'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd',
            'camry', 'land cruiser', 'Royal Enfield Thunder 500',
            'UM Renegade Mojave', 'KTM RC200', 'Bajaj Dominar 400',
            'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R',
            'Royal Enfield Thunder 350', 'KTM 390 Duke ',
            'Mahindra Mojo XT300', 'Bajaj Pulsar RS200',
            'Royal Enfield Bullet 350', 'Royal Enfield Classic 500',
            'Bajaj Avenger 220', 'Bajaj Avenger 150', 'Honda CB Hornet 160R',
            'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160',
            'Bajaj Pulsar 150', 'Honda CBR 150', 'Hero Extreme',
            'Bajaj Avenger 220 dtsi', 'Bajaj Avenger 150 street',
            'Yamaha FZ  v 2.0', 'Bajaj Pulsar  NS 200', 'Bajaj Pulsar 220 F',
            'TVS Apache RTR 180', 'Hero Passion X pro', 'Bajaj Pulsar NS 200',
            'Yamaha Fazer ', 'Honda Activa 4G', 'TVS Sport ',
            'Honda Dream Yuga ', 'Bajaj Avenger Street 220',
            'Hero Splender iSmart', 'Activa 3g', 'Hero Passion Pro',
            'Honda CB Trigger', 'Yamaha FZ S ', 'Bajaj Pulsar 135 LS',
            'Activa 4g', 'Honda CB Unicorn', 'Hero Honda CBZ extreme',
            'Honda Karizma', 'Honda Activa 125', 'TVS Jupyter',
            'Hero Honda Passion Pro', 'Hero Splender Plus', 'Honda CB Shine',
            'Bajaj Discover 100', 'Suzuki Access 125', 'TVS Wego',
            'Honda CB twister', 'Hero Glamour', 'Hero Super Splendor', 'Bajaj Discover 125', 'Hero Hunk', 'Hero  Ignitor Disc',
            'Hero  CBZ Xtreme', 'Bajaj  ct 100', 'i20', 'grand i10', 'i10',
            'eon', 'xcent', 'elantra', 'creta', 'verna', 'city', 'brio',
            'amaze', 'jazz']
            Fuel_categories =['CNG','Diesel','Petrol']
            Seller_categories =['Individual','Dealer']
            Trans_categories =['Automatic','Manual']
            
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[car_categories,Fuel_categories,Seller_categories,Trans_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e: 
            logging.info("Error occured in Data Transformation class")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()
            target_colum_name = 'Present_Price'
            drop_columns = [target_colum_name]

            input_feature_train_df = train_df.drop(columns= drop_columns, axis = 1)
            target_colum_name_train_df = train_df[target_colum_name]

            input_feature_test_df = test_df.drop(columns= drop_columns, axis = 1)
            target_colum_name_test_df = test_df[target_colum_name]

            ## 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying the preprocessing object to train and test datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_colum_name_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_colum_name_test_df)]

            save_function(
                file_path= self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj
            )   

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
    
        except Exception as e:
            logging.info("Error occured in initiate data transformation function")
            raise CustomException(e,sys)