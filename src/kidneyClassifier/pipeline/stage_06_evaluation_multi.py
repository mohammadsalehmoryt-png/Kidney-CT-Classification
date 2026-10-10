import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

class MultiModelEvaluation:
    def __init__(self):
        # المسار الصحيح للمجلد الأساسي (بدون test)
        self.data_dir = "artifacts/data_ingestion/diseased_dataset" 
        self.model_path = "artifacts/training_multi/multi_model.h5"
        self.scores_path = "artifacts/evaluation/multi_scores.json"
        self.cm_path = "artifacts/evaluation/multi_confusion_matrix.png"
        self.classes = ['Cyst', 'Stone', 'Tumor']

    def evaluate(self):
        print(f"جاري تحميل النموذج من: {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)

        # إضافة validation_split ليتطابق مع ما فعلناه في التدريب
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.20
        )
        
        # استخدام subset="validation" لاختبار الـ 20% المخصصة للتحقق
        test_generator = test_datagen.flow_from_directory(
            self.data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False, 
            subset="validation" 
        )

        print("جاري استخراج التنبؤات...")
        predictions = model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # 1. طباعة وحفظ التقرير
        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True)
        print("\n--- تقرير التصنيف ---")
        print(classification_report(y_true, y_pred, target_names=self.classes))

        os.makedirs(os.path.dirname(self.scores_path), exist_ok=True)
        with open(self.scores_path, "w") as f:
            json.dump(report, f, indent=4)

        # 2. رسم مصفوفة الارتباك
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix - Kidney Diseases')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.cm_path)
        print(f"\nتم حفظ مصفوفة الارتباك كصورة في: {self.cm_path}")

if __name__ == '__main__':
    try:
        pipeline = MultiModelEvaluation()
        pipeline.evaluate()
    except Exception as e:
        print(f"حدث خطأ: {e}")