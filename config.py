"""
Конфігураційний модуль для системи розпізнавання об'єктів
"""
import torch

class Config:
    """Глобальна конфігурація системи"""
    
    # Параметри моделі
    MODEL_NAME = 'fasterrcnn_resnet50_fpn'
    PRETRAINED = True
    NUM_CLASSES = 3  # COCO dataset classes
    
    # Параметри детекції
    DETECTION_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Параметри навчання
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    NUM_WORKERS = 2
    
    # Параметри IoU
    IOU_THRESHOLD = 0.5
    
    # Шляхи до даних
    DATA_DIR = './data'
    OUTPUT_DIR = './output'
    CHECKPOINT_DIR = './checkpoints'
    REPORTS_DIR = './reports'
    
    # Параметри візуалізації
    BBOX_COLOR = (0, 255, 0)
    BBOX_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)
    TEXT_SIZE = 0.5
    
    # Пристрій для обчислень
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # COCO класи для візуалізації
    COCO_CLASSES = [
        '__background__', 'person', 'car'
    ]