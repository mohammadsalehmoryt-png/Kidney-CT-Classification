from kidneyClassifier.config.configuration import ConfigurationManager
from kidneyClassifier.components.model_evaluation_binary import ModelEvaluationBinary
from kidneyClassifier.logger import logger
from kidneyClassifier.exception import CustomException
import sys

STAGE_NAME = "Binary Model Evaluation Stage"

class BinaryModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            evaluation = ModelEvaluationBinary(config=eval_config)
            evaluation.evaluation()
            evaluation.save_score()
        except Exception as e:
            logger.error(CustomException(e, sys))
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] بدأت <<<<<<")
        pipeline = BinaryModelEvaluationPipeline()
        pipeline.main()
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] اكتملت بنجاح <<<<<<\n\n======================")
    except Exception as e:
        logger.error(CustomException(e, sys))
        raise e