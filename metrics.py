"""
Модуль обчислення метрик якості детекції об'єктів
"""
import numpy as np
import torch
from typing import List, Dict, Tuple
from utils import calculate_iou
from config import Config


class DetectionMetrics:
    """Клас для обчислення метрик детекції"""
    
    def __init__(self, num_classes: int, iou_threshold: float = None):
        """
        Ініціалізація класу метрик
        
        Args:
            num_classes: Кількість класів
            iou_threshold: Поріг IoU для визначення правильної детекції
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold or Config.IOU_THRESHOLD
    
    def calculate_iou_batch(self, pred_boxes: np.ndarray, 
                           gt_boxes: np.ndarray) -> np.ndarray:
        """
        Обчислення IoU для батчу boxes
        
        Args:
            pred_boxes: Передбачені boxes [N, 4]
            gt_boxes: Ground truth boxes [M, 4]
            
        Returns:
            Матриця IoU [N, M]
        """
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        
        iou_matrix = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                iou_matrix[i, j] = calculate_iou(
                    pred_boxes[i].tolist(),
                    gt_boxes[j].tolist()
                )
        
        return iou_matrix
    
    def calculate_precision_recall(self, predictions: List[Dict], 
                                   ground_truths: List[Dict],
                                   class_id: int) -> Tuple[float, float]:
        """
        Обчислення Precision та Recall для одного класу
        
        Args:
            predictions: Список передбачень
            ground_truths: Список ground truth анотацій
            class_id: ID класу
            
        Returns:
            Tuple (precision, recall)
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            
            gt_boxes = gt['boxes'].cpu().numpy()
            gt_labels = gt['labels'].cpu().numpy()
            
            # Фільтрація по класу
            pred_mask = pred_labels == class_id
            gt_mask = gt_labels == class_id
            
            class_pred_boxes = pred_boxes[pred_mask]
            class_gt_boxes = gt_boxes[gt_mask]
            
            if len(class_pred_boxes) == 0 and len(class_gt_boxes) == 0:
                continue
            
            if len(class_gt_boxes) == 0:
                false_positives += len(class_pred_boxes)
                continue
            
            if len(class_pred_boxes) == 0:
                false_negatives += len(class_gt_boxes)
                continue
            
            # Обчислення IoU матриці
            iou_matrix = self.calculate_iou_batch(class_pred_boxes, class_gt_boxes)
            
            # Визначення TP, FP
            matched_gt = set()
            for i in range(len(class_pred_boxes)):
                max_iou_idx = np.argmax(iou_matrix[i])
                max_iou = iou_matrix[i, max_iou_idx]
                
                if max_iou >= self.iou_threshold and max_iou_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(max_iou_idx)
                else:
                    false_positives += 1
            
            # FN - не збіглися GT boxes
            false_negatives += len(class_gt_boxes) - len(matched_gt)
        
        # Обчислення метрик
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return precision, recall
    
    def calculate_ap(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """
        Обчислення Average Precision (AP)
        
        Args:
            precisions: Масив precision значень
            recalls: Масив recall значень
            
        Returns:
            Average Precision
        """
        # Сортування за recall
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        # Інтерполяція precision
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        
        # Обчислення площі під кривою
        recall_diffs = np.diff(np.concatenate([[0], recalls, [1]]))
        ap = np.sum(precisions * recall_diffs[:-1])
        
        return ap
    
    def calculate_map(self, predictions: List[Dict], 
                     ground_truths: List[Dict],
                     iou_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Обчислення Mean Average Precision (mAP)
        
        Args:
            predictions: Список передбачень
            ground_truths: Список ground truth анотацій
            iou_thresholds: Список порогів IoU
            
        Returns:
            Словник з метриками mAP
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75]
        
        results = {}
        
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            aps = []
            
            # Обчислення AP для кожного класу
            for class_id in range(1, self.num_classes):  # Пропускаємо фон (клас 0)
                precision, recall = self.calculate_precision_recall(
                    predictions, ground_truths, class_id
                )
                
                if precision > 0 or recall > 0:
                    ap = precision  # Спрощена версія AP
                    aps.append(ap)
            
            # mAP для цього порогу
            map_value = np.mean(aps) if aps else 0.0
            results[f'mAP@{iou_thresh}'] = map_value
        
        # Загальний mAP
        results['mAP'] = np.mean([v for v in results.values()])
        
        return results
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Обчислення F1-score
        
        Args:
            precision: Precision
            recall: Recall
            
        Returns:
            F1-score
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_model(self, predictions: List[Dict], 
                      ground_truths: List[Dict]) -> Dict[str, any]:
        """
        Повна оцінка моделі
        
        Args:
            predictions: Список передбачень
            ground_truths: Список ground truth анотацій
            
        Returns:
            Словник з усіма метриками
        """
        results = {
            'per_class': {},
            'overall': {}
        }
        
        # Метрики для кожного класу
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        for class_id in range(1, self.num_classes):
            precision, recall = self.calculate_precision_recall(
                predictions, ground_truths, class_id
            )
            
            if precision > 0 or recall > 0:
                f1 = self.calculate_f1_score(precision, recall)
                
                results['per_class'][class_id] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
        
        # Загальні метрики
        results['overall'] = {
            'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
            'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
            'mean_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0
        }
        
        # mAP метрики
        map_results = self.calculate_map(predictions, ground_truths)
        results['overall'].update(map_results)
        
        return results