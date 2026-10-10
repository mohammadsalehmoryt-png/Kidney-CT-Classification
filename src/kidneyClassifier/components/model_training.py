import tensorflow as tf
from pathlib import Path
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
        """تجهيز محولد البيانات للتدريب والتحقق مع التقسيم والـ Data Augmentation"""
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
        """بدء عملية التدريب وحفظ النموذج عند الاكتمل"""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        logger.info(f"بدء تدريب النموذج الثنائي (Binary Model) بعدد epochs = {self.config.params_epochs}...")
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        logger.info(f"تم تدريب النموذج الأول بنجاح وحفظه في: {self.config.trained_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)