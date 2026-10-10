from kidneyClassifier.config.configuration import ConfigurationManager
from kidneyClassifier.components.prepare_base_model import PrepareBaseModel
from kidneyClassifier.logger import logger
from kidneyClassifier.exception import CustomException
import sys

STAGE_NAME = "Prepare base model stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] بدأت <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] اكتملت بنجاح <<<<<<\n\n======================")
    except Exception as e:
        logger.error(CustomException(e, sys))
        raise e