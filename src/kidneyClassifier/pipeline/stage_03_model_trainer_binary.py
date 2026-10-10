from kidneyClassifier.config.configuration import ConfigurationManager
from kidneyClassifier.components.model_trainer_binary import ModelTrainerBinary
from kidneyClassifier.logger import logger
from kidneyClassifier.exception import CustomException
import sys

STAGE_NAME = "Binary Model Training Stage"

class BinaryModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            training_binary_config = config_manager.get_training_binary_config()
            
            trainer = ModelTrainerBinary(config=training_binary_config)
            trainer.get_base_model()
            trainer.train_valid_generator()
            trainer.train()

        except Exception as e:
            logger.error(CustomException(e, sys))
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] بدأت <<<<<<")
        pipeline = BinaryModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] اكتملت بنجاح <<<<<<\n\n======================")
    except Exception as e:
        logger.error(CustomException(e, sys))
        raise e