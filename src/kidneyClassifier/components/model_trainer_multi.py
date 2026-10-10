import os
import tensorflow as tf
from pathlib import Path
from kidneyClassifier.entity.config_entity import TrainingMultiConfig

class ModelTrainerMulti:
    def __init__(self, config: TrainingMultiConfig):
        self.config = config

    def get_multi_base_model(self):
        # 1. تحميل النموذج الأساسي (VGG16)
        base_model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights='imagenet',
            include_top=False
        )

        # 2. فك التجميد (Fine-Tuning) لآخر كتلة فقط (Block 5)
        for layer in base_model.layers:
            if layer.name.startswith('block5'):
                layer.trainable = True
            else:
                layer.trainable = False

        # 3. بناء الطبقات العميقة مع Regularization (Dropout)
        flatten_in = tf.keras.layers.Flatten()(base_model.output)
        dense_1 = tf.keras.layers.Dense(256, activation='relu')(flatten_in)
        dropout_1 = tf.keras.layers.Dropout(0.5)(dense_1)  # إطفاء 50% من الخلايا لمنع الحفظ
        prediction = tf.keras.layers.Dense(self.config.params_classes, activation='softmax')(dropout_1)

        self.model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=prediction
        )

        # ترجمة النموذج (تأكد من تعديل LEARNING_RATE في params.yaml إلى 0.0001)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        self.model.summary()

    def train_valid_generator(self):
        # إعدادات معالجة الصور ومولدات البيانات (نفس الكود الخاص بك تقريباً)
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical' # categorical لأننا نملك 3 فئات
        )

        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagen.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator_kwargs
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
        # 4. إضافة الـ Callbacks الذكية
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )

        # تقليل معدل التعلم عند توقف التحسن
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=2, 
            min_lr=1e-6,
            verbose=1
        )

        # الإيقاف المبكر لمنع الـ Overfitting واستعادة أفضل أوزان
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=4, 
            restore_best_weights=True,
            verbose=1
        )

        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator,
            callbacks=[checkpoint, reduce_lr, early_stop]
        )