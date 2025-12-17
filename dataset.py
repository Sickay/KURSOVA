"""
Модуль для роботи з датасетом для детекції об'єктів
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import Config
from utils import collate_fn


class ObjectDetectionDataset(Dataset):
    """Dataset для детекції об'єктів"""
    
    def __init__(self, root_dir: str, annotation_file: str = None, 
                 transforms=None, mode: str = 'train'):
        """
        Ініціалізація датасету
        
        Args:
            root_dir: Коренева директорія (train/ або val/)
            annotation_file: Файл з анотаціями (JSON). Якщо None, шукає annotations.json в root_dir
            transforms: Трансформації для зображень
            mode: Режим ('train' або 'val')
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')  # Папка з зображеннями
        self.transforms = transforms
        self.mode = mode
        
        # Визначення шляху до анотацій
        if annotation_file is None:
            annotation_file = os.path.join(root_dir, 'annotations.json')
        
        # Завантаження анотацій
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            print(f"⚠ Файл анотацій не знайдено: {annotation_file}")
            self.annotations = []
        
        # Перевірка існування папки images
        if not os.path.exists(self.images_dir):
            print(f"⚠ Папка зображень не знайдена: {self.images_dir}")
        
        print(f"✓ Завантажено {len(self.annotations)} зразків для режиму '{mode}'")
        print(f"  Папка зображень: {self.images_dir}")
    
    def __len__(self) -> int:
        """Повертає розмір датасету"""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Отримання елементу датасету
        
        Args:
            idx: Індекс елемента
            
        Returns:
            Tuple з зображенням та target словником
        """
        annotation = self.annotations[idx]
        
        # Завантаження зображення з папки images
        img_path = os.path.join(self.images_dir, annotation['image_path'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Зображення не знайдено: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Отримання boxes та labels
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)
        
        # Обчислення площ boxes
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Створення target словника
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Застосування трансформацій
        if self.transforms:
            image_np = self.transforms(image_np)
        
        # Конвертація в тензор
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, target
    
    @staticmethod
    def create_annotation_template(image_paths: List[str], 
                                   output_file: str) -> None:
        """
        Створення шаблону файлу анотацій
        
        Args:
            image_paths: Список шляхів до зображень
            output_file: Вихідний JSON файл
        """
        annotations = []
        
        for img_path in image_paths:
            annotation = {
                'image_path': img_path,
                'boxes': [],  # [[x1, y1, x2, y2], ...]
                'labels': []  # [class_id, ...]
            }
            annotations.append(annotation)
        
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Шаблон анотацій створено: {output_file}")


class DatasetBuilder:
    """Клас для побудови DataLoader'ів"""
    
    @staticmethod
    def build_dataloader(dataset: Dataset, 
                        batch_size: int = None,
                        shuffle: bool = True,
                        num_workers: int = None) -> DataLoader:
        """
        Створення DataLoader
        
        Args:
            dataset: Dataset
            batch_size: Розмір батчу
            shuffle: Перемішувати дані
            num_workers: Кількість робочих процесів
            
        Returns:
            DataLoader
        """
        batch_size = batch_size or Config.BATCH_SIZE
        num_workers = num_workers or Config.NUM_WORKERS
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return dataloader
    
    @staticmethod
    def create_sample_dataset(output_dir: str, num_samples: int = 10) -> None:
        """
        Створення прикладу датасету для тестування
        
        Args:
            output_dir: Директорія для збереження (train/ або val/)
            num_samples: Кількість зразків
        """
        # Створення структури директорій
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        annotations = []
        
        for i in range(num_samples):
            # Створення випадкового зображення
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = f'sample_{i:03d}.jpg'
            full_path = os.path.join(images_dir, img_path)
            
            Image.fromarray(img).save(full_path)
            
            # Створення випадкових анотацій
            num_objects = np.random.randint(1, 5)
            boxes = []
            labels = []
            
            for _ in range(num_objects):
                x1 = np.random.randint(0, 500)
                y1 = np.random.randint(0, 400)
                x2 = x1 + np.random.randint(50, 140)
                y2 = y1 + np.random.randint(50, 80)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(np.random.randint(1, 10))
            
            annotation = {
                'image_path': img_path,
                'boxes': boxes,
                'labels': labels
            }
            annotations.append(annotation)
        
        # Збереження анотацій в root_dir (не в images/)
        annotation_file = os.path.join(output_dir, 'annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"✓ Створено тестовий датасет: {num_samples} зразків")
        print(f"  Структура:")
        print(f"    {output_dir}/")
        print(f"    ├── images/")
        print(f"    │   ├── sample_000.jpg")
        print(f"    │   ├── sample_001.jpg")
        print(f"    │   └── ...")
        print(f"    └── annotations.json")


class COCODatasetAdapter:
    """Адаптер для роботи з COCO форматом"""
    
    @staticmethod
    def convert_coco_to_custom(coco_annotation_file: str, 
                              output_file: str) -> None:
        """
        Конвертація COCO анотацій у власний формат
        
        Args:
            coco_annotation_file: COCO JSON файл
            output_file: Вихідний файл
        """
        with open(coco_annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Створення мапінгу image_id -> annotations
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Конвертація
        custom_annotations = []
        for img in coco_data['images']:
            img_id = img['id']
            
            boxes = []
            labels = []
            
            if img_id in img_to_anns:
                for ann in img_to_anns[img_id]:
                    # COCO format: [x, y, width, height] -> [x1, y1, x2, y2]
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])
            
            custom_annotations.append({
                'image_path': img['file_name'],
                'boxes': boxes,
                'labels': labels
            })
        
        with open(output_file, 'w') as f:
            json.dump(custom_annotations, f, indent=2)
        
        print(f"✓ COCO анотації конвертовано: {output_file}")