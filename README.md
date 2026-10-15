# Kidney CT Scan Classification System

## Overview
نظام برمجي يعتمد على التعلم العميق (Deep Learning) لتحليل وتصنيف صور الأشعة المقطعية للكلى. تم بناء النظام لتوفير أداة تشخيص مساعدة، تعتمد على بنية تصنيف هرمية (Hierarchical Classification).

## System Architecture
* **AI Pipeline:** يعتمد على بنية VGG16 (Transfer Learning) مقسمة إلى مسارين:
  1. تصنيف ثنائي (Binary Classifier): فحص الحالة الأساسية (Normal vs. Diseased).
  2. تصنيف متعدد (Multi-class Classifier): تحديد نوع الإصابة (Tumor, Cyst, Stone).
* **GUI Desktop Application:** واجهة مستخدم رسومية مبرمجة باستخدام إطار عمل PyQt5.
* **Hardware Integration:** مزامنة مع أجهزة الإدخال الخارجي عبر مكتبة OpenCV للاتصال المباشر بكاميرات الفحص أو الأجهزة المتصلة بـ USB.
* **Deployment:** النظام يعمل محلياً بالكامل، ومُجمّع كملف تنفيذي مستقل (.exe) لبيئة ويندوز باستخدام PyInstaller.

## Technologies Used
* Python 3.9+
* TensorFlow / Keras
* PyQt5
* OpenCV
* DVC (Data Version Control)

## Installation & Setup
1. استنساخ المستودع:
   ```bash
   git clone [https://github.com/mohammadsalehmoryt-png/Kidney-CT-Classification.git](https://github.com/mohammadsalehmoryt-png/Kidney-CT-Classification.git)




تثبيت الاعتمادات والمكتبات بالإصدارات المحددة:


python -m pip install -r requirements.txt
Execution






لتشغيل الكود المصدري للواجهة:


python app.py





لعملية البناء والتصدير (Build) لملف ويندوز التنفيذي:


python -m PyInstaller --noconsole --name "KidneyCT_Pro"