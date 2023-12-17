from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    
    dataingestion = DataIngestion()
    dataingestion.load_dataset()

    datatransformation = DataTransformation()
    datatransformation.integrate_data()
    datatransformation.split_data(number_of_test_days = 15)
    datatransformation.transform_data()

    modeltrainer = ModelTrainer()
    modeltrainer.train_model()