"""واجهة عربية محسّنة لنظام تحليل صور الكلى.

تتطلب: PyQt5 و OpenCV. يفترض تكامل النموذج وجود:
src.kidneyClassifier.pipeline.prediction.HierarchicalPredictionPipeline

تنبيه: مخرجات النموذج مساعدة تحليلية وليست تشخيصاً طبياً نهائياً.
"""

import os
import sys
import tempfile
from datetime import datetime

import cv2
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ألوان موحدة للواجهة
COLORS = {
    "background": "#EAF3F7",
    "surface": "#EAF1F7",
    "text": "#172B4D",
    "muted": "#62748A",
    "border": "#E3E8EE",
    "primary": "#1769AA",
    "primary_hover": "#10558D",
    "teal": "#087F75",
    "success": "#16845B",
    "warning": "#A65D00",
    "danger": "#B42318",
}


class FunctionWorker(QObject):
    """تنفيذ مهمة طويلة في QThread حتى لا تتجمد الواجهة."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.finished.emit(self.function())
        except Exception as exc:
            self.failed.emit(str(exc))


class CameraWindow(QDialog):
    """نافذة التقاط صورة من كاميرا الجهاز."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("التقاط صورة من الكاميرا")
        self.setMinimumSize(680, 560)
        self.setLayoutDirection(Qt.RightToLeft)
        self.captured_path = None
        self.cap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        self.video_label = QLabel("جارٍ بدء الكاميرا…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(620, 430)
        self.video_label.setStyleSheet(
            "background:#101820; color:#FFFFFF; border-radius:12px;"
        )
        layout.addWidget(self.video_label, 1)

        controls = QHBoxLayout()
        self.capture_button = QPushButton("التقاط الصورة")
        self.capture_button.setObjectName("primaryButton")
        self.capture_button.clicked.connect(self.capture_image)
        self.cancel_button = QPushButton("إلغاء")
        self.cancel_button.clicked.connect(self.reject)
        controls.addWidget(self.capture_button)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("تعذّر العثور على كاميرا متصلة")
            self.capture_button.setEnabled(False)
            return False
        self.timer.start(33)
        return True

    def update_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def capture_image(self):
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.warning(self, "تعذّر الالتقاط", "لم نتمكن من التقاط إطار من الكاميرا.")
            return
        file_descriptor, path = tempfile.mkstemp(prefix="kidney_scan_", suffix=".jpg")
        os.close(file_descriptor)
        if not cv2.imwrite(path, frame):
            QMessageBox.warning(self, "تعذّر الحفظ", "لم نتمكن من حفظ الصورة الملتقطة.")
            return
        self.captured_path = path
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        event.accept()


class KidneyCTApp(QMainWindow):
    """الواجهة الرئيسية لنظام تحليل صور الكلى."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("نظام تحليل صور الكلى")
        self.setMinimumSize(980, 700)
        self.resize(1360, 880)
        self.setLayoutDirection(Qt.RightToLeft)

        self.pipeline = None
        self.image_path = None
        self.original_pixmap = None
        self.report_text = ""
        self._thread = None
        self._worker = None

        self._apply_style()
        self._build_ui()
        self.setAcceptDrops(True)
        self._set_status("النظام جاهز — يلزم تهيئة محرك التحليل", "neutral")
        self._log("تم فتح الواجهة. هيّئ محرك التحليل، ثم أضف صورة للفحص.")

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {COLORS['background']}; color: {COLORS['text']}; }}
            QLabel {{ color: {COLORS['text']}; }}
            QFrame#card {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 14px; }}
            QFrame#viewer {{ background: #F8FAFC; border: 2px dashed #B8C7D8; border-radius: 14px; }}
            QLabel#pageTitle {{ font-size: 23px; font-weight: 700; }}
            QLabel#subtitle, QLabel#muted {{ color: {COLORS['muted']}; }}
            QLabel#sectionTitle {{ font-size: 15px; font-weight: 700; }}
            QLabel#statusBadge {{ padding: 8px 12px; border-radius: 10px; font-weight: 700; }}
            QPushButton, QToolButton {{ background: white; color: {COLORS['text']}; border: 1px solid {COLORS['border']}; border-radius: 9px; padding: 10px 15px; font-size: 14px; font-weight: 600; }}
            QPushButton:hover, QToolButton:hover {{ background: #EDF4FA; border-color: #A7B9CC; }}
            QPushButton:disabled, QToolButton:disabled {{ background: #E9EEF4; color: #8B98A8; border-color: #D7DFE8; }}
            QPushButton#primaryButton {{ background: {COLORS['primary']}; color: white; border: none; }}
            QPushButton#primaryButton:hover {{ background: {COLORS['primary_hover']}; }}
            QPushButton#secondaryAccent {{ background: {COLORS['teal']}; color: white; border: none; }}
            QPushButton#secondaryAccent:hover {{ background: #06685F; }}
            QPushButton#dangerButton {{ color: {COLORS['danger']}; }}
            QTextEdit {{ background: white; border: 1px solid {COLORS['border']}; border-radius: 9px; padding: 10px; selection-background-color: #B8D9F2; }}
            QProgressBar {{ background: #E6ECF2; border: none; border-radius: 6px; height: 12px; text-align: center; }}
            QProgressBar::chunk {{ background: {COLORS['primary']}; border-radius: 6px; }}
            QMenu {{ background: white; border: 1px solid {COLORS['border']}; padding: 5px; }}
            QMenu::item {{ padding: 9px 22px; }}
            QMenu::item:selected {{ background: #EAF2F9; color: {COLORS['primary']}; }}
        """)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(26, 20, 26, 18)
        main.setSpacing(14)

        # رأس الصفحة
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("تحليل صور الكلى")
        title.setObjectName("pageTitle")
        subtitle = QLabel("مراجعة صورة الأشعة وتشغيل التحليل المساعد")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)

        self.status_badge = QLabel("النظام جاهز")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge)
        self.load_button = QPushButton("تهيئة محرك التحليل")
        self.load_button.setObjectName("primaryButton")
        self.load_button.setToolTip("تحميل نموذج التحليل الطبي في الخلفية")
        self.load_button.clicked.connect(self.load_models)
        header.addWidget(self.load_button)
        main.addLayout(header)

        # شريط خطوات العمل
        steps = QFrame()
        steps.setObjectName("card")
        steps_layout = QHBoxLayout(steps)
        steps_layout.setContentsMargins(18, 10, 18, 10)
        self.step_labels = []
        for text in ("١. إضافة الصورة", "٢. مراجعة الفحص", "٣. نتيجة التحليل"):
            label = QLabel(text)
            label.setObjectName("muted")
            label.setAlignment(Qt.AlignCenter)
            self.step_labels.append(label)
            steps_layout.addWidget(label, 1)
        main.addWidget(steps)

        # منطقة العمل: عارض الصور ولوحة التفاصيل
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        viewer_card = QFrame()
        viewer_card.setObjectName("card")
        viewer_layout = QVBoxLayout(viewer_card)
        viewer_layout.setContentsMargins(16, 14, 16, 16)
        viewer_layout.setSpacing(10)
        viewer_header = QHBoxLayout()
        viewer_title = QLabel("عارض صورة الفحص")
        viewer_title.setObjectName("sectionTitle")
        self.filename_label = QLabel("لم تُضف صورة بعد")
        self.filename_label.setObjectName("muted")
        self.filename_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        viewer_header.addWidget(viewer_title)
        viewer_header.addStretch(1)
        viewer_header.addWidget(self.filename_label)
        viewer_layout.addLayout(viewer_header)

        self.image_frame = QFrame()
        self.image_frame.setObjectName("viewer")
        image_layout = QVBoxLayout(self.image_frame)
        image_layout.setContentsMargins(12, 12, 12, 12)
        self.image_label = QLabel("أضف صورة CT لعرضها هنا\n\nيمكنك اختيار ملف أو سحبه إلى هذه النافذة")
        self.image_label.setObjectName("muted")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(420, 420)
        image_layout.addWidget(self.image_label, 1)
        viewer_layout.addWidget(self.image_frame, 1)

        image_tools = QHBoxLayout()
        self.zoom_out_button = QPushButton("تصغير")
        self.zoom_in_button = QPushButton("تكبير")
        self.fit_button = QPushButton("ملاءمة")
        for button in (self.zoom_out_button, self.zoom_in_button, self.fit_button):
            button.setEnabled(False)
            image_tools.addWidget(button)
        image_tools.addStretch(1)
        viewer_layout.addLayout(image_tools)
        self.zoom_factor = 1.0
        self.zoom_in_button.clicked.connect(lambda: self._change_zoom(1.15))
        self.zoom_out_button.clicked.connect(lambda: self._change_zoom(1 / 1.15))
        self.fit_button.clicked.connect(self._fit_image)
        splitter.addWidget(viewer_card)

        # لوحة بيانات الفحص والنتيجة
        side_card = QFrame()
        side_card.setObjectName("card")
        side_card.setMinimumWidth(285)
        side_card.setMaximumWidth(390)
        side = QVBoxLayout(side_card)
        side.setContentsMargins(18, 16, 18, 16)
        side.setSpacing(12)

        details_title = QLabel("بيانات الفحص")
        details_title.setObjectName("sectionTitle")
        side.addWidget(details_title)
        self.file_info = QLabel("الصورة: —\nالصيغة: —\nالحالة: بانتظار صورة")
        self.file_info.setObjectName("muted")
        self.file_info.setWordWrap(True)
        side.addWidget(self.file_info)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color:{COLORS['border']};")
        side.addWidget(divider)

        result_title = QLabel("ملخص التحليل")
        result_title.setObjectName("sectionTitle")
        side.addWidget(result_title)
        self.result_label = QLabel("ستظهر النتيجة هنا بعد اكتمال التحليل.")
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(78)
        side.addWidget(self.result_label)

        self.confidence_caption = QLabel("مستوى الثقة: —")
        self.confidence_caption.setObjectName("muted")
        side.addWidget(self.confidence_caption)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p٪")
        side.addWidget(self.confidence_bar)

        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        self.result_details.setPlaceholderText("تفاصيل التحليل ستظهر هنا.")
        self.result_details.setMinimumHeight(150)
        side.addWidget(self.result_details, 1)

        caution = QLabel("تنبيه: هذه النتيجة مساعدة تحليلية، ولا تغني عن مراجعة الطبيب المختص.")
        caution.setWordWrap(True)
        caution.setStyleSheet(f"color:{COLORS['warning']}; background:#FFF7E8; padding:10px; border-radius:8px;")
        side.addWidget(caution)
        splitter.addWidget(side_card)
        splitter.setSizes([850, 340])
        main.addWidget(splitter, 1)

        # إجراءات الصفحة: زر واحد لإضافة الصورة وقائمة بالمصادر، ثم الإجراء الأساسي
        actions = QHBoxLayout()
        self.add_image_button = QPushButton("إضافة صورة")
        self.add_image_button.setObjectName("secondaryAccent")
        self.add_image_button.clicked.connect(self.select_image)
        actions.addWidget(self.add_image_button)

        self.source_button = QToolButton()
        self.source_button.setText("مصدر الصورة")
        self.source_button.setPopupMode(QToolButton.InstantPopup)
        source_menu = QMenu(self)
        local_action = QAction("اختيار صورة من الجهاز", self)
        local_action.triggered.connect(self.select_image)
        camera_action = QAction("التقاط صورة بالكاميرا", self)
        camera_action.triggered.connect(self.open_camera)
        source_menu.addAction(local_action)
        source_menu.addAction(camera_action)
        self.source_button.setMenu(source_menu)
        actions.addWidget(self.source_button)
        actions.addStretch(1)

        self.save_button = QPushButton("حفظ التقرير")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_report)
        actions.addWidget(self.save_button)

        self.analyze_button = QPushButton("بدء التحليل")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setMinimumWidth(170)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.run_prediction)
        actions.addWidget(self.analyze_button)
        main.addLayout(actions)

        # شريط حالة وتقدم فعلي، وسجل تقني قابل للطي
        footer = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumWidth(220)
        self.status_text = QLabel("جاهز")
        self.status_text.setObjectName("muted")
        footer.addWidget(self.status_text, 1)
        footer.addWidget(self.progress_bar)
        main.addLayout(footer)

        self.log_toggle = QToolButton()
        self.log_toggle.setText("عرض السجل التقني")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(False)
        self.log_toggle.toggled.connect(self._toggle_log)
        main.addWidget(self.log_toggle, 0, Qt.AlignRight)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(95)
        self.log_text.setLayoutDirection(Qt.RightToLeft)
        self.log_text.hide()
        main.addWidget(self.log_text)

    def _set_status(self, text, kind="neutral"):
        colors = {
            "neutral": ("#EAF0F6", COLORS["text"]),
            "working": ("#E8F2FB", COLORS["primary"]),
            "success": ("#E7F6EF", COLORS["success"]),
            "warning": ("#FFF4DF", COLORS["warning"]),
            "error": ("#FDECEC", COLORS["danger"]),
        }
        background, foreground = colors.get(kind, colors["neutral"])
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"background:{background}; color:{foreground}; padding:8px 12px; border-radius:10px; font-weight:700;"
        )
        self.status_text.setText(text)

    def _log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{stamp}] {message}")

    def _toggle_log(self, checked):
        self.log_text.setVisible(checked)
        self.log_toggle.setText("إخفاء السجل التقني" if checked else "عرض السجل التقني")

    def _start_worker(self, function, on_success):
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = FunctionWorker(function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_success)
        self._worker.failed.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _cleanup_worker(self):
        self._worker = None
        self._thread = None

    def _on_worker_error(self, error_text):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("تعذّرت العملية", "error")
        self._log(f"فشل تنفيذ المهمة: {error_text}")
        self.result_label.setText("تعذّر إكمال العملية. راجع السجل التقني أو أعد المحاولة.")
        self.load_button.setEnabled(self.pipeline is None)
        self.add_image_button.setEnabled(True)
        self.source_button.setEnabled(True)
        self.analyze_button.setEnabled(bool(self.pipeline and self.image_path))
        QMessageBox.critical(self, "تعذّرت العملية", error_text)

    def load_models(self):
        if self.pipeline is not None:
            return
        self.load_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self._set_status("جارٍ تهيئة محرك التحليل…", "working")
        self.result_label.setText("يتم تحميل النموذج في الخلفية، ويمكنك متابعة حالة العملية هنا.")
        self._log("بدء تهيئة محرك التحليل.")

        def initialize_pipeline():
            from src.kidneyClassifier.pipeline.prediction import HierarchicalPredictionPipeline
            return HierarchicalPredictionPipeline()

        self._start_worker(initialize_pipeline, self._on_models_loaded)

    def _on_models_loaded(self, pipeline):
        self.pipeline = pipeline
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.load_button.setText("محرك التحليل جاهز")
        self.load_button.setEnabled(False)
        self.add_image_button.setEnabled(True)
        self.source_button.setEnabled(True)
        self._set_status("محرك التحليل جاهز", "success")
        self._log("تمت تهيئة محرك التحليل بنجاح.")
        self._update_steps(1)
        if self.image_path:
            self.analyze_button.setEnabled(True)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "اختيار صورة الفحص",
            "",
            "صور طبية مدعومة (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;كل الملفات (*)",
        )
        if path:
            self.set_image_for_analysis(path)

    def open_camera(self):
        dialog = CameraWindow(self)
        if not dialog.start_camera():
            QMessageBox.warning(self, "الكاميرا غير متاحة", "تحقق من اتصال الكاميرا وصلاحيات استخدامها.")
            dialog.close()
            return
        if dialog.exec_() == QDialog.Accepted and dialog.captured_path:
            self.set_image_for_analysis(dialog.captured_path)
        else:
            self._log("أُلغي التقاط الصورة.")

    def set_image_for_analysis(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "صيغة غير مدعومة",
                "تعذّر فتح الملف كصورة. عارض DICOM غير مدمج في هذه النسخة؛ حوّل الملف إلى PNG أو JPEG أولاً.",
            )
            return
        self.image_path = path
        self.original_pixmap = pixmap
        self.zoom_factor = 1.0
        self.filename_label.setText(os.path.basename(path))
        self.file_info.setText(
            f"الصورة: {os.path.basename(path)}\nالصيغة: {os.path.splitext(path)[1].upper().lstrip('.') or 'غير معروفة'}\n"
            f"الأبعاد: {pixmap.width()} × {pixmap.height()} بكسل\nالحالة: جاهزة للمراجعة"
        )
        self.zoom_in_button.setEnabled(True)
        self.zoom_out_button.setEnabled(True)
        self.fit_button.setEnabled(True)
        self._fit_image()
        self.result_label.setText("تم تحميل الصورة. راجعها ثم ابدأ التحليل.")
        self.result_details.clear()
        self.confidence_caption.setText("مستوى الثقة: —")
        self.confidence_bar.setValue(0)
        self.save_button.setEnabled(False)
        self.analyze_button.setEnabled(self.pipeline is not None)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_status("تم تحميل الصورة — جاهزة للمراجعة", "success")
        self._log(f"تم تحميل الصورة: {path}")
        self._update_steps(2 if self.pipeline else 1)

    def _fit_image(self):
        if self.original_pixmap is None:
            return
            
        width = max(100, self.image_label.width() - 20)
        height = max(100, self.image_label.height() - 20)
        
        scaled = self.original_pixmap.scaled(
            int(width * self.zoom_factor),
            int(height * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")
    
    def _change_zoom(self, factor):
        if self.original_pixmap is None:
            return
        self.zoom_factor = min(4.0, max(0.25, self.zoom_factor * factor))
        # التكبير هنا نسبة إلى المساحة المتاحة في العارض.
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap is not None:
            self._fit_image()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path and os.path.isfile(path):
            self.set_image_for_analysis(path)
            event.acceptProposedAction()

    def run_prediction(self):
        if not self.pipeline or not self.image_path:
            return
        pipeline = self.pipeline
        image_path = self.image_path
        self.analyze_button.setEnabled(False)
        self.add_image_button.setEnabled(False)
        self.source_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self._set_status("جارٍ تحليل الصورة…", "working")
        self.result_label.setText("يعالج النظام الصورة الآن. قد يستغرق ذلك بعض الوقت.")
        self.result_details.clear()
        self._log(f"بدأ تحليل الصورة: {os.path.basename(image_path)}")
        self._update_steps(2)

        self._start_worker(lambda: pipeline.predict(image_path), self._on_prediction_finished)

    @staticmethod
    def _confidence_percent(value):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        # بعض النماذج تُرجع قيمة بين 0 و1، وأخرى نسبة مئوية.
        if 0 <= confidence <= 1:
            confidence *= 100
        return min(100.0, max(0.0, confidence))

    def _on_prediction_finished(self, result):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.add_image_button.setEnabled(True)
        self.source_button.setEnabled(True)

        if not isinstance(result, dict):
            raise ValueError("أعاد النموذج نتيجة غير متوقعة؛ كان المتوقع قاموساً.")

        # استخراج نسب الثقة للنموذجين
        confidence = self._confidence_percent(result.get("Confidence", 0))
        
        # معالجة ثقة البوابة الثنائية (النموذج الأول)
        binary_conf_raw = str(result.get("Binary_Confidence", "0")).replace('%', '')
        binary_confidence = self._confidence_percent(binary_conf_raw)
        # إذا لم يُرجع النظام ثقة للثنائي (في حالة السليم)، نستخدم الثقة العامة
        if binary_confidence == 0.0:
            binary_confidence = confidence 

        raw_status = str(result.get("Final_Diagnosis", "غير محدد")).replace("Diseased:", "").strip()
        names = {
            "Normal": "كلى سليمة (Normal)",
            "Cyst": "كيس مائي في الكلى (Cyst)",
            "Tumor": "ورم كلوي (Tumor)",
            "Stone": "حصوة كلوية (Stone)",
        }
        diagnosis = names.get(raw_status, raw_status or "نتيجة غير محددة")
        route_taken = result.get('Route_Taken', 'غير متاح')

        # ==========================================
        # بناء التقرير الهرمي المُفصّل
        # ==========================================
        details = " تفاصيل التحليل (Hierarchical Pipeline):\n"

        if "Normal" in raw_status:
            # حالة الكلى السليمة (يعمل النموذج الأول فقط)
            summary = f"🟢 النتيجة: {diagnosis}"
            kind = "success"
            
            details += " المرحلة الأولى: البوابة الثنائية (Binary Classifier)\n"
            details += " التشخيص: لم يتم اكتشاف أنسجة شاذة (Normal).\n"
            details += f" ثقة النموذج الأول: {binary_confidence:.1f}%\n\n"
            
            details += " المرحلة الثانية: المصنف الدقيق (Multi-Class Classifier)\n"
            details += "⏸ تم إيقاف المسار: لا حاجة لتفعيل النموذج الثاني نظراً لسلامة الكلى.\n\n"
            
        elif confidence < 70:
            # حالة الشك أو الثقة المنخفضة
            summary = " لم نستطع تشخيص حالة الكلى بدقة عالية."
            kind = "warning"
            
            details += " المرحلة الأولى: البوابة الثنائية (Binary Classifier)\n"
            details += " التشخيص: تم اكتشاف أنسجة غير طبيعية (Diseased).\n"
            details += f" ثقة النموذج الأول: {binary_confidence:.1f}%\n\n"
            
            details += " المرحلة الثانية: المصنف الدقيق (Multi-Class Classifier)\n"
            details += f" الاشتباه المبدئي: {diagnosis}\n"
            details += f" ثقة النموذج الثاني: {confidence:.1f}% (منخفضة جداً!)\n\n"
            details += "الرجاء مراجعة طبيب مختص لإجراء فحص سريري نظراً لتدني نسبة الثقة.\n\n"
            
        else:
            # حالة الإصابة المؤكدة (يعمل النموذجان معاً)
            summary = f"🔴 إصابة مكتشفة: {diagnosis}"
            kind = "warning"
            
            details += "▶ المرحلة الأولى: البوابة الثنائية (Binary Classifier)\n"
            details += " التشخيص: تم اكتشاف أنسجة غير طبيعية (Diseased).\n"
            details += f" ثقة النموذج الأول: {binary_confidence:.1f}%\n\n"
            
            details += " المرحلة الثانية: المصنف الدقيق (Multi-Class Classifier)\n"
            details += f" نوع الإصابة: {diagnosis}\n"
            details += f" ثقة النموذج الثاني: {confidence:.1f}%\n\n"

        details += f"🔗 المسار الكلي للشبكة العصبية:\n[ {route_taken} ]\n"

        # ==========================================
        # تحديث الواجهة
        # ==========================================
        self.result_label.setText(summary)
        self.result_label.setStyleSheet(
            f"font-weight:700; color:{COLORS['success'] if kind == 'success' else COLORS['warning']};"
        )
        self.confidence_caption.setText(f"مستوى الثقة النهائي: {confidence:.1f}٪")
        self.confidence_bar.setValue(round(confidence))
        
        self.result_details.setPlainText(details)
        self.report_text = details
        
        self.save_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self._set_status("اكتمل التحليل", kind)
        self._log("اكتمل التحليل. تم عرض تفاصيل المسار الهرمي بنجاح.")
        self._update_steps(3)

    def save_report(self):
        if not self.report_text:
            return
        default_name = f"تقرير_تحليل_الكلى_{datetime.now():%Y%m%d_%H%M}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "حفظ التقرير", default_name, "ملف نصي (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as report_file:
                report_file.write("تقرير مساعدة تحليلية لصور الكلى\n")
                report_file.write(f"التاريخ: {datetime.now():%Y-%m-%d %H:%M}\n")
                report_file.write(f"الصورة: {os.path.basename(self.image_path or '')}\n\n")
                report_file.write(self.report_text)
                report_file.write("\n\nتنبيه: لا يغني هذا التقرير عن مراجعة الطبيب المختص.\n")
            self._set_status("تم حفظ التقرير", "success")
            self._log(f"تم حفظ التقرير في: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "تعذّر حفظ التقرير", str(exc))

    def _update_steps(self, active_step):
        for index, label in enumerate(self.step_labels, start=1):
            if index < active_step:
                label.setStyleSheet(f"color:{COLORS['success']}; font-weight:700;")
            elif index == active_step:
                label.setStyleSheet(f"color:{COLORS['primary']}; font-weight:700;")
            else:
                label.setStyleSheet(f"color:{COLORS['muted']};")


def main():
    # إضافة مجلد المشروع إلى المسار حتى يعمل استيراد خط التحليل.
    project_root = os.getcwd()
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setLayoutDirection(Qt.RightToLeft)
    app.setFont(QFont("Noto Sans Arabic", 10))
    window = KidneyCTApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
