from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    binary_dataset_dir: Path
    diseased_dataset_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_classes: int

@dataclass(frozen=True)
class TrainingBinaryConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data_dir: Path
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_is_augmentation: bool
    params_classes: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_data: Path
    evaluated_scores_path: Path
    params_image_size: list
    params_batch_size: int

@dataclass(frozen=True)
class TrainingMultiConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data_dir: Path
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_is_augmentation: bool
    params_classes: int