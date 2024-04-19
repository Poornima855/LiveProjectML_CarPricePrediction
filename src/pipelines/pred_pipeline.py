import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd
from src.utils import Project_dir_path

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join(Project_dir_path(), 'artifacts\\preprocessor.pkl')
            model_path = os.path.join(Project_dir_path(), "artifacts\\model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, Car_Name:str, 
                     Selling_Price:float, 
                     Kms_Driven:int, 
                     Owner:int, 
                     Year:int,
                     Fuel_Type:str, 
                     Seller_Type:str, 
                     Transmission:str): 
             self.Car_Name = Car_Name
             self.Selling_Price = Selling_Price
             self.Kms_Driven = Kms_Driven
             self.Owner = Owner
             self.Year = Year
             self.Fuel_Type = Fuel_Type
             self.Seller_Type = Seller_Type
             self.Transmission = Transmission
        
        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = {
                       'Car_Name': [self.Car_Name], 
                       'Selling_Price': [self.Selling_Price], 
                       'Kms_Driven': [self.Kms_Driven], 
                       'Owner': [self.Owner],
                       'Year':[self.Year],
                       'Fuel_Type':[self.Fuel_Type], 
                       'Seller_Type': [self.Seller_Type], 
                       'Transmission': [self.Transmission]

                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")
                  return df
             except Exception as e:
                  logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                  raise CustomException(e,sys) 
             
             
        