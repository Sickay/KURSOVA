"""
Модуль моделі Faster R-CNN для детекції об'єктів
"""
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import List, Dict, Any, Optional
from config import Config


class FasterRCNNDetector:
    """Клас для роботи з моделлю Faster R-CNN"""
    
    def __init__(self, num_classes: Optional[int] = None, pretrained: bool = True):
        """
        Ініціалізація детектора
        
        Args:
            num_classes: Кількість класів (включаючи фон)
            pretrained: Завантажити попередньо навчену модель
        """
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.device = Config.DEVICE
        self.model = self._build_model(pretrained)
        self.model.to(self.device)
        
        print(f"✓ Модель Faster R-CNN ініціалізована на {self.device}")
        print(f"  Кількість класів: {self.num_classes}")
    
    def _build_model(self, pretrained: bool) -> torch.nn.Module:
        """
        Побудова моделі Faster R-CNN
        
        Args:
            pretrained: Використовувати попередньо навчені ваги
            
        Returns:
            Модель Faster R-CNN
        """
        # Завантаження базової моделі
        if pretrained:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=self.num_classes)
            return model
        
        # Заміна голови для власної кількості класів
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
    
    def predict(self, images: List[torch.Tensor], 
                threshold: float = None) -> List[Dict[str, torch.Tensor]]:
        """
        Виконання детекції об'єктів
        
        Args:
            images: Список тензорів зображень
            threshold: Поріг впевненості
            
        Returns:
            Список словників з результатами детекції
        """
        threshold = threshold or Config.DETECTION_THRESHOLD
        
        self.model.eval()
        with torch.no_grad():
            # Переміщення зображень на пристрій
            images = [img.to(self.device) for img in images]
            
            # Детекція
            predictions = self.model(images)
            
            # Фільтрація за порогом впевненості
            filtered_predictions = []
            for pred in predictions:
                mask = pred['scores'] > threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }
                filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def train_mode(self):
        """Перемикання моделі в режим навчання"""
        self.model.train()
    
    def eval_mode(self):
        """Перемикання моделі в режим оцінки"""
        self.model.eval()
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None):
        """
        Збереження чекпоінту моделі
        
        Args:
            path: Шлях для збереження
            epoch: Номер епохи
            optimizer_state: Стан оптимізатора
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        print(f"✓ Чекпоінт збережено: {path}")
    
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Завантаження чекпоінту моделі
        
        Args:
            path: Шлях до чекпоінту
            optimizer: Оптимізатор для завантаження стану
            
        Returns:
            Номер епохи з чекпоінту
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Чекпоінт завантажено: {path} (епоха {epoch})")
        
        return epoch
    
    def get_model_parameters(self):
        """Отримання параметрів моделі для оптимізатора"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def count_parameters(self) -> int:
        """
        Підрахунок кількості параметрів моделі
        
        Returns:
            Загальна кількість параметрів
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Прямий прохід через модель
        
        Args:
            images: Список зображень
            targets: Список цільових значень (для навчання)
            
        Returns:
            Словник з втратами (навчання) або предикції (інференс)
        """
        images = [img.to(self.device) for img in images]
        
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            return self.model(images, targets)
        else:
            return self.model(images)