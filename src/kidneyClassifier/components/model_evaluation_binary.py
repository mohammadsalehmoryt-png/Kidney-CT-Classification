import tensorflow as tf
from pathlib import Path
from kidneyClassifier.entity.config_entity import EvaluationConfig
from kidneyClassifier.utils.common import save_json
from kidneyClassifier.logger import logger

class ModelEvaluationBinary:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        """تجهيز مولد بيانات التحقق لتقييم النموذج بدقة"""
        datagen_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluation(self):
        """تحميل النموذج وتقييمه باستخدام مجموعة التحقق"""
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        """حفظ النتائج النهائية في ملف JSON"""
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=self.config.evaluated_scores_path, data=scores)
        logger.info(f"تم حفظ مقاييس التقييم في: {self.config.evaluated_scores_path}")
        logger.info(f"النتيجة النهائية للنموذج الأول - Loss: {self.score[0]:.4f}, Accuracy: {self.score[1]*100:.2f}%")