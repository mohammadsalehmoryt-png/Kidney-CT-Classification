import os
import sys
import logging
from datetime import datetime

# تسمية ملف السجل برقم التاريخ والساعة لضمان عدم التداخل
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# إنشاء مجلد logs في بيئة العمل الرئيسية
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# إعداد السجلات وكتابتها في الملف وعلى الشاشة (Console)
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("kidneyClassifier")