"""
Модуль навчання та донавчання моделі Faster R-CNN
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from config import Config
from model import FasterRCNNDetector
from metrics import DetectionMetrics
import matplotlib.pyplot as plt


class ModelTrainer:
    """Клас для навчання моделі детекції"""
    
    def __init__(self, model: FasterRCNNDetector, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = None):
        """
        Ініціалізація тренера
        
        Args:
            model: Модель для навчання
            train_loader: DataLoader для тренувальних даних
            val_loader: DataLoader для валідаційних даних
            learning_rate: Швидкість навчання
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE
        
        # Оптимізатор
        self.lr = learning_rate or Config.LEARNING_RATE
        self.optimizer = torch.optim.SGD(
            self.model.get_model_parameters(),
            lr=self.lr,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )
        
        # Історія навчання
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print("✓ Тренер ініціалізовано")
        print(f"  Швидкість навчання: {self.lr}")
        print(f"  Розмір тренувального набору: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Розмір валідаційного набору: {len(val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Навчання на одній епосі
        
        Args:
            epoch: Номер епохи
            
        Returns:
            Середнє значення втрат
        """
        self.model.train_mode()
        
        total_loss = 0
        num_batches = 0
        
        print(f"\n{'='*60}")
        print(f"Епоха {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            print(f"\n➡️ START BATCH {batch_idx}")
            # Переміщення на пристрій
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Прямий прохід
            loss_dict = self.model.forward(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Зворотне поширення
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            
            # Статистика
            total_loss += losses.item()
            num_batches += 1
            
            # Логування
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] | "
                      f"Loss: {losses.item():.4f} | Avg Loss: {avg_loss:.4f}")
        
        epoch_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"\n  Середні втрати епохи: {epoch_loss:.4f}")
        print(f"  Час епохи: {epoch_time:.2f}s")
        
        return epoch_loss
    
    def validate(self) -> float:
        """
        Валідація моделі
        
        Returns:
            Середнє значення втрат на валідації
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval_mode()
        
        total_loss = 0
        num_batches = 0
        
        print("\n  Валідація...")
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model.forward(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches
        print(f"  Втрати валідації: {val_loss:.4f}")
        
        return val_loss
    
    def train(self, num_epochs: int = None, 
              save_checkpoint: bool = True) -> Dict[str, List[float]]:
        """
        Повний цикл навчання
        
        Args:
            num_epochs: Кількість епох
            save_checkpoint: Зберігати чекпоінти
            
        Returns:
            Історія навчання
        """
        num_epochs = num_epochs or Config.NUM_EPOCHS
        
        print("\n" + "="*60)
        print("ПОЧАТОК НАВЧАННЯ")
        print("="*60)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Навчання
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Валідація
            val_loss = self.validate()
            if val_loss > 0:
                self.history['val_loss'].append(val_loss)
            
            # Збереження LR
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Scheduler step
            self.scheduler.step()
            
            # Збереження чекпоінту
            if save_checkpoint and ((epoch + 1) % 5 == 0):
                checkpoint_path = os.path.join(
                    Config.CHECKPOINT_DIR, 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                self.model.save_checkpoint(
                    checkpoint_path, 
                    epoch, 
                    self.optimizer.state_dict()
                )
                
                # Збереження найкращої моделі
                if val_loss > 0 and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(
                        Config.CHECKPOINT_DIR, 
                        'best_model.pth'
                    )
                    self.model.save_checkpoint(
                        best_model_path, 
                        epoch, 
                        self.optimizer.state_dict()
                    )
                    print(f"  ★ Збережено найкращу модель (val_loss: {val_loss:.4f})")
        
        print("\n" + "="*60)
        print("НАВЧАННЯ ЗАВЕРШЕНО")
        print("="*60)
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Побудова графіків навчання
        
        Args:
            save_path: Шлях для збереження графіків
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Графік втрат
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        if self.history['val_loss']:
            axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        axes[0].set_xlabel('Епоха')
        axes[0].set_ylabel('Втрати')
        axes[0].set_title('Історія навчання - Втрати')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Графік швидкості навчання
        axes[1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_xlabel('Епоха')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Швидкість навчання')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Графіки навчання збережено: {save_path}")
        
        plt.show()
    
    def fine_tune(self, num_epochs: int = 5, 
                  freeze_backbone: bool = True) -> Dict[str, List[float]]:
        """
        Донавчання моделі на власному датасеті
        
        Args:
            num_epochs: Кількість епох донавчання
            freeze_backbone: Заморозити backbone
            
        Returns:
            Історія донавчання
        """
        print("\n" + "="*60)
        print("ПОЧАТОК ДОНАВЧАННЯ")
        print("="*60)
        
        # Заморозка backbone якщо потрібно
        if freeze_backbone:
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False
            print("  ✓ Backbone заморожено")
        
        # Зменшена швидкість навчання для донавчання
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * 0.1
        
        print(f"  ✓ Швидкість навчання зменшено до {self.lr * 0.1}")
        
        # Навчання
        return self.train(num_epochs=num_epochs, save_checkpoint=True)