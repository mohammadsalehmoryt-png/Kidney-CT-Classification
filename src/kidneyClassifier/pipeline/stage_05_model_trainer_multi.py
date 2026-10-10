from kidneyClassifier.config.configuration import ConfigurationManager
from kidneyClassifier.components.model_trainer_multi import ModelTrainerMulti
from kidneyClassifier.logger import logger
from kidneyClassifier.exception import CustomException
import sys

STAGE_NAME = "Multi-class Model Training Stage"

class MultiModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            training_config = config.get_training_multi_config()
            trainer = ModelTrainerMulti(config=training_config)
            trainer.get_multi_base_model()
            trainer.train_valid_generator()
            trainer.train()
        except Exception as e:
            logger.error(CustomException(e, sys))
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] بدأت <<<<<<")
        pipeline = MultiModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] اكتملت بنجاح <<<<<<\n\n======================")
    except Exception as e:
        logger.error(CustomException(e, sys))
        raise e