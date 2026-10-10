from pathlib import Path
from kidneyClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from kidneyClassifier.utils.common import read_yaml, create_directories
from kidneyClassifier.entity.config_entity import (
    DataIngestionConfig, 
    PrepareBaseModelConfig, 
    TrainingBinaryConfig,
    EvaluationConfig,
    TrainingMultiConfig  # تأكد من استيراد الكيان الجديد
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        # 1. قراءة ملفات الإعدادات والمعلمات Centralized Config & Params
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # 2. إنشاء المجلد الرئيسي للمخرجات (artifacts)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """جلب إعدادات مرحلة فحص واستيراد البيانات"""
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            binary_dataset_dir=Path(config.binary_dataset_dir),
            disease_dataset_dir=Path(config.disease_dataset_dir)
        )

        return data_ingestion_config

  
# أضف هذه الدالة داخل كلاس ConfigurationManager:
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation
        create_directories([Path(eval_config.root_dir)])

        evaluation_config = EvaluationConfig(
            path_of_model=Path(eval_config.path_of_model),
            training_data=Path(self.config.data_ingestion.binary_dataset_dir),
            all_data=Path(eval_config.all_data_dir),
            evaluated_scores_path=Path(Path(eval_config.root_dir) / "scores.json"),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return evaluation_config
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """جلب إعدادات مرحلة تجهيز النموذج الأساسي (VGG16)"""
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights="imagenet",
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    def get_training_multi_config(self) -> TrainingMultiConfig:
        training = self.config.training_multi
        params = self.params
        create_directories([Path(training.root_dir)])

        training_multi_config = TrainingMultiConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(training.updated_base_model_path),
            training_data_dir=Path(training.training_data_dir),
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_epochs=params.EPOCHS,
            params_learning_rate=params.LEARNING_RATE,
            params_is_augmentation=params.AUGMENTATION,
            params_classes=params.CLASSES
        )

        return training_multi_config

    def get_training_binary_config(self) -> TrainingBinaryConfig:
        """جلب إعدادات مرحلة تدريب النموذج الأول (Binary Classification)"""
        training = self.config.training_binary
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        create_directories([Path(training.root_dir)])

        training_binary_config = TrainingBinaryConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data_dir=Path(self.config.data_ingestion.binary_dataset_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_binary_config