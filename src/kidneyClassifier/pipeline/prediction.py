import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# --- حل مشكلة استهلاك الذاكرة (Memory Growth) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# -----------------------------------------------------------------

class HierarchicalPredictionPipeline:
    def __init__(self): # 1. أزلنا filename من هنا
        
        # التأكد من وجود مسارات النماذج
        binary_model_path = "artifacts/training_binary/binary_model.h5"
        multi_model_path = "artifacts/training_multi/multi_model.h5"
        
        if not os.path.exists(binary_model_path):
            raise FileNotFoundError(f"النموذج الثنائي غير موجود في: {binary_model_path}")
        if not os.path.exists(multi_model_path):
            raise FileNotFoundError(f"النموذج المتعدد غير موجود في: {multi_model_path}")

        # 2. تحميل النماذج (تحدث مرة واحدة فقط عند بدء التطبيق)
        print("جاري تحميل النماذج وتهيئة الذكاء الاصطناعي... (الرجاء الانتظار بضع ثوانٍ)")
        self.binary_model = tf.keras.models.load_model(binary_model_path)
        self.multi_model = tf.keras.models.load_model(multi_model_path)

        # تعيين الفئات 
        self.binary_classes = {0: 'Diseased', 1: 'Normal'} 
        self.multi_classes = {0: 'Cyst', 1: 'Stone', 2: 'Tumor'}
        print("اكتمل التحميل! النظام جاهز للفحص السريع.")

    def predict(self, filename): # 3. أضفنا filename هنا
        try:
            print(f"جاري معالجة الصورة: {filename}")
            
            # المعالجة المسبقة للصورة
            img = image.load_img(filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # التطبيع

            # المرحلة الأولى: فحص السلامة (Binary Gatekeeper)
            binary_pred = self.binary_model.predict(img_array)
            binary_idx = np.argmax(binary_pred, axis=1)[0]
            binary_result = self.binary_classes[binary_idx]
            binary_confidence = float(np.max(binary_pred))

            # مسار التوجيه (Routing) بناءً على النتيجة
            if binary_result == 'Normal':
                return {
                    "Status": "Success",
                    "Final_Diagnosis": "Normal Kidney", 
                    "Confidence": round(binary_confidence * 100, 2),
                    "Route_Taken": "Stage 1 Only (Binary)",
                    "Message": "الكلى سليمة، لا حاجة للمرحلة الثانية."
                }
            else:
                # الكلى مصابة -> تحويل للنموذج المتعدد
                multi_pred = self.multi_model.predict(img_array)
                multi_idx = np.argmax(multi_pred, axis=1)[0]
                disease_result = self.multi_classes[multi_idx]
                multi_confidence = float(np.max(multi_pred))

                return {
                    "Status": "Success",
                    "Final_Diagnosis": f"Diseased: {disease_result}",
                    "Confidence": round(multi_confidence * 100, 2),
                    "Binary_Confidence": round(binary_confidence * 100, 2),
                    "Route_Taken": "Stage 1 (Diseased) -> Stage 2 (Multi-class)",
                    "Message": f"تم اكتشاف إصابة وتصنيفها كـ {disease_result}."
                }
        except Exception as e:
            return {"Status": "Error", "Message": str(e)}

# لاختبار السكريبت مباشرة في الطرفية:
if __name__ == '__main__':
    sample_image = "artifacts/data_ingestion/binary_dataset/Diseased/Cyst- (3).jpg"    
    if os.path.exists(sample_image):
        # 4. لاحظ كيف أصبح الاستدعاء هنا (التحميل أولاً، ثم التنبؤ)
        pipeline = HierarchicalPredictionPipeline() # تحميل النماذج
        result = pipeline.predict(sample_image) # فحص الصورة
        
        print("\n--- نتيجة الفحص النهائي ---")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(f"صورة الاختبار غير موجودة في المسار: {sample_image}")