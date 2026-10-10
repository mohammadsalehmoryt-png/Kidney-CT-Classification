import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.utils import class_weight
from kidneyClassifier.entity.config_entity import TrainingBinaryConfig
from kidneyClassifier.logger import logger

class ModelTrainerBinary:
    def __init__(self, config: TrainingBinaryConfig):
        self.config = config

    def get_base_model(self):
        """تحميل النموذج الأساسي المحدث وتجميع المحسن واللوس"""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        # إعادة تجميع النموذج للتأكد من ربطه بمعدل التعلم واللوس الصحيح
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """تجهيز مولد البيانات للتدريب والتحقق مع التقسيم والـ Data Augmentation"""
        datagen_kwargs = dict(
            rescale=1./255,
            validation_split=0.20  # 20% للتحقق و 80% للتدريب
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
            directory=self.config.training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagen_kwargs
            )
        else:
            train_datagen = valid_datagen

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def train(self):
        """بدء عملية التدريب مع معالجة عدم توازن البيانات وحفظ أفضل نموذج تلقائياً"""
        # حساب أوزان الفئات تلقائياً لمعالجة عدم التوازن بين Normal و Diseased
        classes_labels = self.train_generator.classes
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(classes_labels),
            y=classes_labels
        )
        class_weights_dict = dict(enumerate(weights))
        logger.info(f"أوزان الفئات المحسوبة لمعالجة عدم التوازن: {class_weights_dict}")

        logger.info(f"بدء تدريب النموذج الثنائي (Binary Model) بعدد epochs = {self.config.params_epochs}...")
        
        # إعداد الحفظ التلقائي لأفضل نموذج بناءً على أعلى val_accuracy
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            class_weight=class_weights_dict,
            callbacks=[checkpoint_callback]
        )

        logger.info(f"تم الانتهاء من التدريب وحفظ أفضل نموذج في: {self.config.trained_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)