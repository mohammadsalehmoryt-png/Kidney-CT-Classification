import os
import json
import yaml
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from kidneyClassifier.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """قراءة ملف YAML وإرجاع كائن ConfigBox للوصول للمتغيرات باستخدام (.)"""
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"تم تحميل ملف yaml بنجاح من: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("ملف yaml فارغ")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """إنشاء قائمة المجلدات على القرص الصلب"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"تم إنشاء المجلد في: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """حفظ البيانات في ملف JSON"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"تم حفظ ملف json في: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """تحميل ملف JSON وإرجاعه كـ ConfigBox"""
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"تم تحميل ملف json بنجاح من: {path}")
    return ConfigBox(content)


@ensure_annotations
def get_size(path: Path) -> str:
    """إرجاع حجم الملف بالكيلوبايت"""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"