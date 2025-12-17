"""
Модуль допоміжних функцій для обробки зображень та даних
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from config import Config


def load_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Завантаження та підготовка зображення
    
    Args:
        image_path: Шлях до зображення
        
    Returns:
        Tuple з оригінальним зображенням та тензором для моделі
    """
    # Завантаження зображення
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не вдалося завантажити зображення: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Конвертація для PyTorch
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    
    return image, image_tensor


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Збереження зображення
    
    Args:
        image: Зображення для збереження
        output_path: Шлях для збереження
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    """
    Зміна розміру зображення зі збереженням пропорцій
    
    Args:
        image: Вхідне зображення
        target_size: Цільовий розмір
        
    Returns:
        Змінене зображення
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized


def apply_transforms(image: np.ndarray, augment: bool = False) -> np.ndarray:
    """
    Застосування трансформацій до зображення
    
    Args:
        image: Вхідне зображення
        augment: Застосовувати аугментацію
        
    Returns:
        Трансформоване зображення
    """
    if augment:
        # Випадкова зміна яскравості
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Випадкове відзеркалення
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
    
    return image


def create_directory_structure() -> None:
    """Створення структури директорій для проєкту"""
    directories = [
        Config.DATA_DIR,
        Config.OUTPUT_DIR,
        Config.CHECKPOINT_DIR,
        Config.REPORTS_DIR,
        os.path.join(Config.OUTPUT_DIR, 'visualizations'),
        os.path.join(Config.OUTPUT_DIR, 'predictions')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Структура директорій створена")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Обчислення Intersection over Union (IoU) між двома bounding boxes
    
    Args:
        box1: Перший bbox [x1, y1, x2, y2]
        box2: Другий bbox [x1, y1, x2, y2]
        
    Returns:
        Значення IoU (0-1)
    """
    # Координати перетину
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Площа перетину
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Площі bbox'ів
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Площа об'єднання
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       threshold: float = 0.4) -> List[int]:
    """
    Non-Maximum Suppression для фільтрації bounding boxes
    
    Args:
        boxes: Масив bbox'ів
        scores: Оцінки впевненості
        threshold: Поріг IoU
        
    Returns:
        Індекси відібраних bbox'ів
    """
    if len(boxes) == 0:
        return []
    
    # Сортування за оцінками
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Обчислення IoU з іншими boxes
        ious = np.array([calculate_iou(boxes[current], boxes[i]) 
                        for i in indices[1:]])
        
        # Видалення boxes з високим IoU
        indices = indices[1:][ious < threshold]
    
    return keep


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Функція для об'єднання батчу даних
    
    Args:
        batch: Список елементів батчу
        
    Returns:
        Кортеж з зображеннями та таргетами
    """
    return tuple(zip(*batch))