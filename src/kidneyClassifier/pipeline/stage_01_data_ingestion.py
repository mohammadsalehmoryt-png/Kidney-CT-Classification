from kidneyClassifier.config.configuration import ConfigurationManager
from kidneyClassifier.components.data_ingestion import DataIngestion
from kidneyClassifier.logger import logger
import sys
from kidneyClassifier.exception import CustomException

STAGE_NAME = "Data Ingestion Stage (Manual Setup Verification)"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # 1. تهيئة مدير الإعدادات
            config_manager = ConfigurationManager()
            
            # 2. جلب إعدادات جلب البيانات
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            # 3. تهيئة مكون جلب البيانات
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            # 4. تشغيل عملية التحقق
            data_ingestion.verify_datasets()

        except Exception as e:
            logger.error(CustomException(e, sys))
            raise e

# اختبار السكريبت إذا تم تشغيله مباشرة
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] بدأت <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> مرحلة [{STAGE_NAME}] اكتملت بنجاح <<<<<<\n\n======================")
    except Exception as e:
        logger.error(CustomException(e, sys))
        raise e