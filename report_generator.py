"""
Модуль автоматичної генерації звітів про роботу системи
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from config import Config


class ReportGenerator:
    """Клас для генерації звітів"""
    
    def __init__(self, output_dir: str = None):
        """
        Ініціалізація генератора звітів
        
        Args:
            output_dir: Директорія для збереження звітів
        """
        self.output_dir = output_dir or Config.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {},
            'training_info': {},
            'evaluation_metrics': {},
            'detection_results': []
        }
    
    def add_model_info(self, info: Dict) -> None:
        """
        Додавання інформації про модель
        
        Args:
            info: Словник з інформацією про модель
        """
        self.report_data['model_info'].update(info)
    
    def add_training_info(self, history: Dict) -> None:
        """
        Додавання інформації про навчання
        
        Args:
            history: Історія навчання
        """
        self.report_data['training_info'] = {
            'num_epochs': len(history.get('train_loss', [])),
            'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'best_val_loss': min(history['val_loss']) if history.get('val_loss') else None,
            'training_history': history
        }
    
    def add_evaluation_metrics(self, metrics: Dict) -> None:
        """
        Додавання метрик оцінки
        
        Args:
            metrics: Словник з метриками
        """
        self.report_data['evaluation_metrics'].update(metrics)
    
    def add_detection_result(self, image_name: str, 
                           num_detections: int,
                           classes_detected: List[str]) -> None:
        """
        Додавання результату детекції
        
        Args:
            image_name: Назва зображення
            num_detections: Кількість детекцій
            classes_detected: Список виявлених класів
        """
        self.report_data['detection_results'].append({
            'image': image_name,
            'num_detections': num_detections,
            'classes': classes_detected
        })
    
    def generate_text_report(self, filename: str = 'report.txt') -> str:
        """
        Генерація текстового звіту
        
        Args:
            filename: Назва файлу звіту
            
        Returns:
            Шлях до збереженого звіту
        """
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ЗВІТ ПРО РОБОТУ СИСТЕМИ РОЗПІЗНАВАННЯ ОБ'ЄКТІВ\n")
            f.write("="*70 + "\n\n")
            
            # Загальна інформація
            f.write(f"Дата створення: {self.report_data['timestamp']}\n\n")
            
            # Інформація про модель
            if self.report_data['model_info']:
                f.write("-"*70 + "\n")
                f.write("ІНФОРМАЦІЯ ПРО МОДЕЛЬ\n")
                f.write("-"*70 + "\n")
                for key, value in self.report_data['model_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Інформація про навчання
            if self.report_data['training_info']:
                f.write("-"*70 + "\n")
                f.write("ІНФОРМАЦІЯ ПРО НАВЧАННЯ\n")
                f.write("-"*70 + "\n")
                train_info = self.report_data['training_info']
                
                if train_info.get('num_epochs'):
                    f.write(f"Кількість епох: {train_info['num_epochs']}\n")
                if train_info.get('final_train_loss'):
                    f.write(f"Фінальні втрати (train): {train_info['final_train_loss']:.4f}\n")
                if train_info.get('final_val_loss'):
                    f.write(f"Фінальні втрати (val): {train_info['final_val_loss']:.4f}\n")
                if train_info.get('best_val_loss'):
                    f.write(f"Найкращі втрати (val): {train_info['best_val_loss']:.4f}\n")
                f.write("\n")
            
            # Метрики оцінки
            if self.report_data['evaluation_metrics']:
                f.write("-"*70 + "\n")
                f.write("МЕТРИКИ ОЦІНКИ МОДЕЛІ\n")
                f.write("-"*70 + "\n")
                
                metrics = self.report_data['evaluation_metrics']
                
                if 'overall' in metrics:
                    f.write("\nЗагальні метрики:\n")
                    for key, value in metrics['overall'].items():
                        f.write(f"  {key}: {value:.4f}\n")
                
                if 'per_class' in metrics:
                    f.write("\nМетрики по класах:\n")
                    for class_id, class_metrics in metrics['per_class'].items():
                        f.write(f"\n  Клас {class_id}:\n")
                        for key, value in class_metrics.items():
                            f.write(f"    {key}: {value:.4f}\n")
                f.write("\n")
            
            # Результати детекції
            if self.report_data['detection_results']:
                f.write("-"*70 + "\n")
                f.write("РЕЗУЛЬТАТИ ДЕТЕКЦІЇ\n")
                f.write("-"*70 + "\n")
                f.write(f"Загальна кількість зображень: {len(self.report_data['detection_results'])}\n\n")
                
                for result in self.report_data['detection_results']:
                    f.write(f"Зображення: {result['image']}\n")
                    f.write(f"  Кількість детекцій: {result['num_detections']}\n")
                    if result['classes']:
                        f.write(f"  Виявлені класи: {', '.join(result['classes'])}\n")
                    f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("КІНЕЦЬ ЗВІТУ\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Текстовий звіт збережено: {report_path}")
        return report_path
    
    def generate_json_report(self, filename: str = 'report.json') -> str:
        """
        Генерація JSON звіту
        
        Args:
            filename: Назва файлу звіту
            
        Returns:
            Шлях до збереженого звіту
        """
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON звіт збережено: {report_path}")
        return report_path
    
    def generate_visual_report(self, filename: str = 'visual_report.png') -> str:
        """
        Генерація візуального звіту з графіками
        
        Args:
            filename: Назва файлу звіту
            
        Returns:
            Шлях до збереженого звіту
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Сітка для графіків
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Історія навчання - втрати
        if 'training_history' in self.report_data['training_info']:
            ax1 = fig.add_subplot(gs[0, :])
            history = self.report_data['training_info']['training_history']
            
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if history.get('val_loss'):
                ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            
            ax1.set_xlabel('Епоха', fontsize=12)
            ax1.set_ylabel('Втрати', fontsize=12)
            ax1.set_title('Історія навчання моделі', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # 2. Метрики по класах
        if 'per_class' in self.report_data['evaluation_metrics']:
            ax2 = fig.add_subplot(gs[1, 0])
            per_class = self.report_data['evaluation_metrics']['per_class']
            
            classes = list(per_class.keys())
            precisions = [per_class[c]['precision'] for c in classes]
            recalls = [per_class[c]['recall'] for c in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax2.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
            ax2.bar(x + width/2, recalls, width, label='Recall', color='coral')
            
            ax2.set_xlabel('Клас', fontsize=12)
            ax2.set_ylabel('Значення', fontsize=12)
            ax2.set_title('Precision і Recall по класах', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Загальні метрики
        if 'overall' in self.report_data['evaluation_metrics']:
            ax3 = fig.add_subplot(gs[1, 1])
            overall = self.report_data['evaluation_metrics']['overall']
            
            metrics_names = list(overall.keys())
            metrics_values = list(overall.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
            ax3.barh(metrics_names, metrics_values, color=colors)
            
            ax3.set_xlabel('Значення', fontsize=12)
            ax3.set_title('Загальні метрики моделі', fontsize=14, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Додавання значень на графік
            for i, v in enumerate(metrics_values):
                ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        # 4. Статистика детекцій
        if self.report_data['detection_results']:
            ax4 = fig.add_subplot(gs[2, :])
            
            num_detections = [r['num_detections'] for r in self.report_data['detection_results']]
            
            ax4.hist(num_detections, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Кількість детекцій на зображення', fontsize=12)
            ax4.set_ylabel('Частота', fontsize=12)
            ax4.set_title('Розподіл кількості детекцій', fontsize=14, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Статистика
            mean_det = np.mean(num_detections)
            ax4.axvline(mean_det, color='red', linestyle='--', 
                       label=f'Середнє: {mean_det:.1f}')
            ax4.legend(fontsize=10)
        
        report_path = os.path.join(self.output_dir, filename)
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Візуальний звіт збережено: {report_path}")
        return report_path
    
    def generate_full_report(self) -> Dict[str, str]:
        """
        Генерація повного звіту (текст, JSON та візуалізація)
        
        Returns:
            Словник з шляхами до згенерованих звітів
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        paths = {
            'text': self.generate_text_report(f'report_{timestamp}.txt'),
            'json': self.generate_json_report(f'report_{timestamp}.json'),
            'visual': self.generate_visual_report(f'visual_report_{timestamp}.png')
        }
        
        print("\n" + "="*70)
        print("✓ ПОВНИЙ ЗВІТ ЗГЕНЕРОВАНО")
        print("="*70)
        for report_type, path in paths.items():
            print(f"  {report_type.upper()}: {path}")
        print("="*70 + "\n")
        
        return paths