"""
Модуль візуалізації результатів детекції об'єктів
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional
import torch
from config import Config


class DetectionVisualizer:
    """Клас для візуалізації результатів детекції"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Ініціалізація візуалізатора
        
        Args:
            class_names: Список назв класів
        """
        self.class_names = class_names or Config.COCO_CLASSES
        self.colors = self._generate_colors(len(self.class_names))
    
    def _generate_colors(self, num_classes: int) -> Dict[int, tuple]:
        """
        Генерація кольорів для класів
        
        Args:
            num_classes: Кількість класів
            
        Returns:
            Словник з кольорами для кожного класу
        """
        np.random.seed(42)
        colors = {}
        for i in range(num_classes):
            colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors
    
    def draw_boxes(self, image: np.ndarray, 
                   predictions: Dict[str, torch.Tensor],
                   show_scores: bool = True,
                   show_labels: bool = True) -> np.ndarray:
        """
        Малювання bounding boxes на зображенні
        
        Args:
            image: Вхідне зображення
            predictions: Словник з передбаченнями
            show_scores: Показувати оцінки впевненості
            show_labels: Показувати мітки класів
            
        Returns:
            Зображення з намальованими boxes
        """
        image_copy = image.copy()
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            
            # Колір для класу
            color = self.colors.get(label, (0, 255, 0))
            
            # Малювання прямокутника
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 
                         Config.BBOX_THICKNESS)
            
            # Підготовка тексту
            label_text = ""
            if show_labels and label < len(self.class_names):
                label_text = self.class_names[label]
            if show_scores:
                score_text = f"{score:.2f}"
                label_text = f"{label_text} {score_text}" if label_text else score_text
            
            # Малювання тексту
            if label_text:
                # Фон для тексту
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                    Config.TEXT_SIZE, 1
                )
                
                cv2.rectangle(image_copy, 
                            (x1, y1 - text_height - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Текст
                cv2.putText(image_copy, label_text, 
                           (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           Config.TEXT_SIZE,
                           (255, 255, 255), 1)
        
        return image_copy
    
    def visualize_predictions(self, image: np.ndarray,
                            predictions: Dict[str, torch.Tensor],
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
        """
        Візуалізація результатів детекції з matplotlib
        
        Args:
            image: Вхідне зображення
            predictions: Словник з передбаченнями
            save_path: Шлях для збереження
            show: Показати графік
        """
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            
            # Колір для класу
            color = np.array(self.colors.get(label, (0, 255, 0))) / 255.0
            
            # Прямокутник
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Текст
            label_name = self.class_names[label] if label < len(self.class_names) else f"Class {label}"
            text = f"{label_name}: {score:.2f}"
            ax.text(x1, y1 - 5, text, color='white', fontsize=10,
                   bbox=dict(facecolor=color, alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Візуалізацію збережено: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_detection_statistics(self, predictions: List[Dict[str, torch.Tensor]],
                                 save_path: Optional[str] = None) -> None:
        """
        Побудова статистики детекцій
        
        Args:
            predictions: Список передбачень
            save_path: Шлях для збереження
        """
        # Збір статистики
        class_counts = {}
        all_scores = []
        
        for pred in predictions:
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            all_scores.extend(scores)
            
            for label in labels:
                class_name = self.class_names[label] if label < len(self.class_names) else f"Class {label}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Створення графіків
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Розподіл класів
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            axes[0].bar(range(len(classes)), counts, color='skyblue')
            axes[0].set_xticks(range(len(classes)))
            axes[0].set_xticklabels(classes, rotation=45, ha='right')
            axes[0].set_xlabel('Класи')
            axes[0].set_ylabel('Кількість детекцій')
            axes[0].set_title('Розподіл детекцій по класах')
            axes[0].grid(axis='y', alpha=0.3)
        
        # Розподіл оцінок впевненості
        if all_scores:
            axes[1].hist(all_scores, bins=20, color='coral', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Оцінка впевненості')
            axes[1].set_ylabel('Частота')
            axes[1].set_title('Розподіл оцінок впевненості')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].axvline(x=Config.DETECTION_THRESHOLD, color='red', 
                          linestyle='--', label=f'Поріг: {Config.DETECTION_THRESHOLD}')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Статистику збережено: {save_path}")
        
        plt.show()