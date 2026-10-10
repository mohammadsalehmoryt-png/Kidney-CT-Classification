import os
from kidneyClassifier.logger import logger
from kidneyClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def verify_datasets(self):
        """
        التحقق من وجود المجلدات التي تم إنشاؤها
        وإحصاء عدد الصور بداخلها للتأكد من سلامة البيانات
        """
        logger.info("بدء التحقق من وجود البيانات المُجهزة يدوياً...")

        # 1. التحقق من Binary Dataset
        if os.path.exists(self.config.binary_dataset_dir):
            binary_classes = os.listdir(self.config.binary_dataset_dir)
            logger.info(f"مجلد binary_dataset موجود. الفئات: {binary_classes}")
            for cls in binary_classes:
                cls_path = os.path.join(self.config.binary_dataset_dir, cls)
                if os.path.isdir(cls_path):
                    num_images = len(os.listdir(cls_path))
                    logger.info(f" - فئة [{cls}] تحتوي على: {num_images} صورة.")
        else:
            logger.error(f"لم يتم العثور على مجلد binary_dataset في المسار: {self.config.binary_dataset_dir}")
            raise FileNotFoundError("مجلد binary_dataset مفقود!")

        # 2. التحقق من Disease Dataset
        if os.path.exists(self.config.disease_dataset_dir):
            disease_classes = os.listdir(self.config.disease_dataset_dir)
            logger.info(f"مجلد disease_dataset موجود. الفئات: {disease_classes}")
            for cls in disease_classes:
                cls_path = os.path.join(self.config.disease_dataset_dir, cls)
                if os.path.isdir(cls_path):
                    num_images = len(os.listdir(cls_path))
                    logger.info(f" - فئة [{cls}] تحتوي على: {num_images} صورة.")
        else:
            logger.error(f"لم يتم العثور على مجلد disease_dataset في المسار: {self.config.disease_dataset_dir}")
            raise FileNotFoundError("مجلد disease_dataset مفقود!")

        logger.info("تم التحقق من البيانات بنجاح، وهي جاهزة للمرحلة القادمة.")