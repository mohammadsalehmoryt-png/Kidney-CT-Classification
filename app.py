import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QTextEdit, QProgressBar, QFrame, QGraphicsDropShadowEffect, 
                             QMenu, QAction, QDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QColor, QImage
from PyQt5.QtCore import Qt, QTimer

from src.kidneyClassifier.pipeline.prediction import HierarchicalPredictionPipeline

# ==========================================
# نافذة الكاميرا المخصصة (Camera Dialog)
# ==========================================
class CameraWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("التقاط صورة من الجهاز")
        self.setFixedSize(640, 580)
        self.setStyleSheet("""
            QDialog { background-color: #121212; }
            QLabel { color: #00E676; font-weight: bold; }
            QPushButton { background-color: #1E1E1E; color: white; border: 1px solid #00E676; border-radius: 10px; padding: 10px; font-size: 14px; }
            QPushButton:hover { background-color: #00E676; color: black; }
            QPushButton#StopBtn { border: 1px solid #F44336; }
            QPushButton#StopBtn:hover { background-color: #F44336; color: white; }
        """)

        self.captured_path = None
        self.cap = None

        layout = QVBoxLayout(self)
        
        self.video_label = QLabel("جاري تشغيل الكاميرا...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px dashed #333; background-color: black;")
        self.video_label.setFixedSize(600, 450)
        layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 التقاط الصورة")
        self.btn_capture.clicked.connect(self.capture_image)
        
        self.btn_stop = QPushButton("❌ إيقاف وإلغاء")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("❌ لم يتم العثور على كاميرا متصلة!")
            self.btn_capture.setEnabled(False)
            return
        self.timer.start(30) # تحديث الإطار كل 30 ملي ثانية

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # تحويل ألوان أوبن سي في إلى ألوان بايثون الرسومية
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_path = "captured_scan.jpg"
            cv2.imwrite(self.captured_path, frame)
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

# ==========================================
# الواجهة الرئيسية للتطبيق
# ==========================================
class KidneyCTAppPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("نظام تشخيص أمراض الكلى الذكي - الإصدار الاحترافي")
        self.setGeometry(100, 100, 900, 800)
        
        # --- تصميم عصري ---
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel#TitleLabel { color: #00E676; font-family: 'Segoe UI', Arial; }
            QLabel#SubTitleLabel { color: #B0BEC5; }
            QFrame#ImageFrame { background-color: #1E1E1E; border: 2px dashed #00E676; border-radius: 15px; }
            
            QPushButton { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #00E676; border-radius: 20px; padding: 12px 20px; font-size: 15px; font-weight: bold; }
            QPushButton:hover { background-color: #00E676; color: #121212; }
            QPushButton:disabled { background-color: #2C2C2C; border: 1px solid #424242; color: #757575; }
            
            QPushButton#AnalyzeBtn { background-color: #2962FF; border: none; }
            QPushButton#AnalyzeBtn:hover { background-color: #448AFF; }
            QPushButton#AnalyzeBtn:disabled { background-color: #2C2C2C; color: #757575; }
            
            QPushButton#DeviceBtn { background-color: #9C27B0; border: none; }
            QPushButton#DeviceBtn:hover { background-color: #BA68C8; }
            QPushButton#DeviceBtn::menu-indicator { image: none; } /* إخفاء سهم القائمة الافتراضي */
            
            QMenu { background-color: #1E1E1E; color: white; border: 1px solid #9C27B0; font-size: 14px; padding: 5px; }
            QMenu::item { padding: 10px 30px; }
            QMenu::item:selected { background-color: #9C27B0; color: white; border-radius: 5px; }
            
            QTextEdit { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 10px; padding: 15px; font-size: 15px; color: #E0E0E0; font-family: 'Consolas', monospace; }
            QProgressBar { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 8px; text-align: center; color: white; height: 15px; }
            QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #00E676, stop:1 #2962FF); border-radius: 6px; }
        """)

        self.image_path = None
        self.pipeline = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)

        title = QLabel("AI Kidney CT Scan Diagnostics")
        title.setObjectName("TitleLabel")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("نظام التحليل الهرمي الذكي لصور الأشعة المقطعية")
        subtitle.setObjectName("SubTitleLabel")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        self.image_frame = QFrame()
        self.image_frame.setObjectName("ImageFrame")
        self.image_frame.setFixedSize(450, 450)
        image_layout = QVBoxLayout(self.image_frame)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 230, 118, 50))
        shadow.setOffset(0, 0)
        self.image_frame.setGraphicsEffect(shadow)

        self.image_label = QLabel("اسحب الصورة أو استخدم الأزرار أدناه")
        self.image_label.setObjectName("SubTitleLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 14))
        image_layout.addWidget(self.image_label)
        
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(self.image_frame)
        center_layout.addStretch()
        main_layout.addLayout(center_layout)

        # الأزرار
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.btn_load_ai = QPushButton("تهيئة محرك الذكاء الاصطناعي")
        self.btn_load_ai.setObjectName("AnalyzeBtn")
        self.btn_load_ai.setCursor(Qt.PointingHandCursor)
        self.btn_load_ai.clicked.connect(self.load_models)
        
        # زر الأجهزة مع القائمة المنسدلة (Dropdown Menu)
        self.btn_device = QPushButton("الأجهزة المتصلة 📡")
        self.btn_device.setObjectName("DeviceBtn")
        self.btn_device.setCursor(Qt.PointingHandCursor)
        self.btn_device.setEnabled(False) 
        
        # إعداد القائمة المنسدلة
        self.device_menu = QMenu(self)
        
        action_camera = QAction("📷 التقاط من كاميرا الفحص", self)
        action_camera.triggered.connect(self.open_camera)
        
        action_ct = QAction("🩻 استيراد من جهاز الأشعة المقطعية (CT)", self)
        action_ct.triggered.connect(self.import_from_ct_scanner)
        
        action_bt = QAction("📱 استقبال صورة من الهاتف (بلوتوث)", self)
        action_bt.triggered.connect(self.receive_bluetooth)

        self.device_menu.addAction(action_camera)
        self.device_menu.addAction(action_ct)
        self.device_menu.addAction(action_bt)
        self.btn_device.setMenu(self.device_menu)

        self.btn_upload = QPushButton("رفع ملف محلي (Upload)")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.setEnabled(False)
        self.btn_upload.clicked.connect(self.select_image)
        
        self.btn_analyze = QPushButton("بدء التحليل (Analyze)")
        self.btn_analyze.setObjectName("AnalyzeBtn")
        self.btn_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.run_prediction)
        
        button_layout.addStretch()
        button_layout.addWidget(self.btn_load_ai)
        button_layout.addWidget(self.btn_device)
        button_layout.addWidget(self.btn_upload)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100) 
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(140)
        self.log_text.setLayoutDirection(Qt.RightToLeft)
        main_layout.addWidget(self.log_text)

        self.update_log("> الواجهة جاهزة. اضغط على 'تهيئة محرك الذكاء الاصطناعي' للبدء.")

    def load_models(self):
        self.btn_load_ai.setEnabled(False)
        self.update_log("> جاري تحميل النماذج الطبية في الذاكرة (سيتجمد التطبيق لثوانٍ قليلة)...")
        QApplication.processEvents() 
        
        try:
            self.pipeline = HierarchicalPredictionPipeline()
            self.btn_load_ai.hide() 
            self.btn_upload.setEnabled(True)
            self.btn_device.setEnabled(True) # تفعيل زر الأجهزة
            self.progress_bar.setValue(100)
            self.update_log("> ✅ تمت تهيئة محرك الذكاء الاصطناعي بنجاح.\n> اختر مصدر الصورة للبدء (رفع، كاميرا، بلوتوث، أو CT).")
        except Exception as e:
            self.update_log(f"> ❌ حدث خطأ أثناء التهيئة:\n{str(e)}")
            self.btn_load_ai.setEnabled(True)

    # ==========================================
    # وظائف الأجهزة المختلفة
    # ==========================================
    def open_camera(self):
        self.update_log("> 📷 جاري تشغيل الكاميرا...")
        cam_dialog = CameraWindow(self)
        cam_dialog.start_camera()
        cam_dialog.exec_() # إيقاف النافذة الرئيسية حتى ينتهي من الكاميرا
        
        if cam_dialog.captured_path:
            self.set_image_for_analysis(cam_dialog.captured_path)
        else:
            self.update_log("> تم إلغاء التقاط الصورة من الكاميرا.")

    def import_from_ct_scanner(self):
        self.update_log("> 🩻 جاري الاتصال بخادم جهاز الأشعة المقطعية (CT/DICOM Server)...")
        # هنا يمكنك مستقبلاً ربطه بمسار شبكة المستشفى (Network Drive). حالياً سنفتح نافذة اختيار ملفات للمحاكاة.
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "استيراد من خادم الأشعة المقطعية", "", "Images (*.png *.jpg *.jpeg *.dcm)", options=options)
        if file_name:
            self.set_image_for_analysis(file_name)

    def receive_bluetooth(self):
        self.update_log("> 📱 جاري تفعيل البلوتوث بانتظار استلام صورة من هاتف الطبيب...")
        # تنبيه محاكاة لعملية البلوتوث (يمكن ربطه بـ socket بلوتوث حقيقي لاحقاً)
        QMessageBox.information(self, "نقل عبر البلوتوث", "تم فتح منفذ البلوتوث.\nيرجى إرسال الصورة من هاتفك إلى هذا الحاسوب، ثم اختيارها يدوياً كالتالي لحين ربط الخادم.")
        self.select_image()

    def select_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "اختر صورة مقطعية", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.set_image_for_analysis(file_name)

    def set_image_for_analysis(self, file_path):
        self.image_path = file_path
        pixmap = QPixmap(self.image_path)
        self.image_label.clear()
        self.image_label.setPixmap(pixmap.scaled(430, 430, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setValue(0)
        self.update_log(f"> تم إدراج الصورة بنجاح: {os.path.basename(self.image_path)}\n> النظام جاهز. اضغط 'بدء التحليل'.")

    # ==========================================
    # وظائف التحليل
    # ==========================================
    def run_prediction(self):
        self.btn_upload.setEnabled(False)
        self.btn_device.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.update_log("> 🔍 جاري تحليل الأنسجة واستخراج السمات العميقة...")
        self.progress_bar.setValue(50)
        QApplication.processEvents()
        
        try:
            result = self.pipeline.predict(self.image_path)
            self.on_prediction_finished(result)
        except Exception as e:
            self.update_log(f"> ❌ خطأ في النظام:\n{str(e)}")
            self.btn_upload.setEnabled(True)
            self.btn_device.setEnabled(True)
            self.progress_bar.setValue(0)

    def on_prediction_finished(self, result):
        self.progress_bar.setValue(100)
        report = "--- التقرير الطبي للذكاء الاصطناعي ---\n"
        status = result.get('Final_Diagnosis', '')
        
        if "Normal" in status:
            report += "> 🟢 النتيجة: كلى سليمة (Normal)\n"
            report += f"> مستوى الثقة: {result.get('Confidence')}%\n"
        else:
            report += f"> 🔴 النتيجة: إصابة مكتشفة - {status.replace('Diseased: ', '')}\n"
            report += f"> ثقة البوابة الثنائية: {result.get('Binary_Confidence')}%\n"
            report += f"> ثقة المصنف الفرعي: {result.get('Confidence')}%\n"
            
        report += f"\n> مسار التحليل: {result.get('Route_Taken')}"
        
        self.update_log(report)
        self.btn_upload.setEnabled(True)
        self.btn_device.setEnabled(True)

    def update_log(self, text):
        self.log_text.setText(text)

if __name__ == '__main__':
    os.environ['PYTHONPATH'] = os.getcwd() + '/src'
    sys.path.append(os.getcwd() + '/src')
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = KidneyCTAppPro()
    window.show()
    sys.exit(app.exec_())