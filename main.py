"""
Головний модуль системи розпізнавання об'єктів на базі Faster R-CNN
"""
import os
import argparse
import torch
from config import Config
from utils import create_directory_structure, load_image, save_image
from model import FasterRCNNDetector
from visualizer import DetectionVisualizer
from metrics import DetectionMetrics
from dataset import ObjectDetectionDataset, DatasetBuilder
from trainer import ModelTrainer
from report_generator import ReportGenerator


def parse_arguments():
    """Парсинг аргументів командного рядка"""
    parser = argparse.ArgumentParser(
        description='Система розпізнавання об\'єктів з Faster R-CNN'
    )
    
    parser.add_argument('--mode', type=str, default='detect',
                       choices=['detect', 'train', 'evaluate', 'demo'],
                       help='Режим роботи програми')
    
    parser.add_argument('--image', type=str, default=None,
                       help='Шлях до зображення для детекції')
    
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR,
                       help='Директорія з даними')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Шлях до чекпоінту моделі')
    
    parser.add_argument('--threshold', type=float, default=Config.DETECTION_THRESHOLD,
                       help='Поріг впевненості для детекції')
    
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Кількість епох навчання')
    
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Розмір батчу')
    
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Швидкість навчання')
    
    parser.add_argument('--num-classes', type=int, default=Config.NUM_CLASSES,
                       help='Кількість класів')
    
    return parser.parse_args()


def detect_objects(image_path: str, model: FasterRCNNDetector, 
                   visualizer: DetectionVisualizer, threshold: float):
    """
    Детекція об'єктів на зображенні
    
    Args:
        image_path: Шлях до зображення
        model: Модель детектора
        visualizer: Візуалізатор
        threshold: Поріг впевненості
    """
    print(f"\n{'='*60}")
    print(f"ДЕТЕКЦІЯ ОБ'ЄКТІВ")
    print(f"{'='*60}")
    print(f"Зображення: {image_path}")
    
    # Завантаження зображення
    image, image_tensor = load_image(image_path)
    print(f"✓ Зображення завантажено: {image.shape}")
    
    # Детекція
    predictions = model.predict([image_tensor], threshold=threshold)
    pred = predictions[0]
    
    num_detections = len(pred['boxes'])
    print(f"✓ Виявлено об'єктів: {num_detections}")
    
    if num_detections > 0:
        # Виведення детекцій
        for i, (box, label, score) in enumerate(zip(
            pred['boxes'].cpu().numpy(),
            pred['labels'].cpu().numpy(),
            pred['scores'].cpu().numpy()
        )):
            class_name = Config.COCO_CLASSES[label] if label < len(Config.COCO_CLASSES) else f"Class {label}"
            print(f"  {i+1}. {class_name}: {score:.3f} | Box: {box.astype(int).tolist()}")
        
        # Візуалізація
        image_with_boxes = visualizer.draw_boxes(image, pred)
        
        # Збереження
        output_path = os.path.join(Config.OUTPUT_DIR, 'visualizations', 
                                   os.path.basename(image_path))
        save_image(image_with_boxes, output_path)
        
        # Matplotlib візуалізація
        viz_path = os.path.join(Config.OUTPUT_DIR, 'visualizations',
                               f"viz_{os.path.basename(image_path)}")
        visualizer.visualize_predictions(image, pred, save_path=viz_path, show=False)
    else:
        print("  Об'єкти не виявлено")
    
    print(f"{'='*60}\n")
    
    return predictions


def train_model(args):
    """
    Навчання моделі
    
    Args:
        args: Аргументи командного рядка
    """
    print(f"\n{'='*60}")
    print(f"НАВЧАННЯ МОДЕЛІ")
    print(f"{'='*60}\n")
    
    # Створення тестового датасету якщо не існує
    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        print("⚠ Тренувальний датасет не знайдено. Створюємо тестовий датасет...")
        DatasetBuilder.create_sample_dataset(train_dir, num_samples=20)
    
    # Завантаження датасетів
    train_dataset = ObjectDetectionDataset(
        root_dir=train_dir,
        annotation_file=None,  # Автоматично шукає annotations.json в train_dir
        mode='train'
    )
    
    train_loader = DatasetBuilder.build_dataloader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Ініціалізація моделі
    model = FasterRCNNDetector(num_classes=args.num_classes, pretrained=True)
    
    # Завантаження чекпоінту якщо вказано
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_checkpoint(args.checkpoint)
    
    # Тренер
    trainer = ModelTrainer(model, train_loader, learning_rate=args.lr)
    
    # Навчання
    history = trainer.train(num_epochs=args.epochs, save_checkpoint=True)
    
    # Графіки
    plot_path = os.path.join(Config.OUTPUT_DIR, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)
    
    # Генерація звіту
    report_gen = ReportGenerator()
    report_gen.add_model_info({
        'model_name': 'Faster R-CNN ResNet50-FPN',
        'num_classes': args.num_classes,
        'num_parameters': model.count_parameters(),
        'device': str(Config.DEVICE)
    })
    report_gen.add_training_info(history)
    report_gen.generate_full_report()
    
    print("\n✓ Навчання завершено успішно!")


def evaluate_model(args):
    """
    Оцінка моделі
    
    Args:
        args: Аргументи командного рядка
    """
    print(f"\n{'='*60}")
    print(f"ОЦІНКА МОДЕЛІ")
    print(f"{'='*60}\n")
    
    # Завантаження моделі
    model = FasterRCNNDetector(num_classes=args.num_classes, pretrained=True)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_checkpoint(args.checkpoint)
    else:
        print("⚠ Використовується попередньо навчена модель")
    
    # Завантаження валідаційного датасету
    val_dir = os.path.join(args.data_dir, 'val')
    if not os.path.exists(val_dir):
        print("⚠ Валідаційний датасет не знайдено. Створюємо тестовий датасет...")
        DatasetBuilder.create_sample_dataset(val_dir, num_samples=10)
    
    val_dataset = ObjectDetectionDataset(
        root_dir=val_dir,
        annotation_file=None,  # Автоматично шукає annotations.json в val_dir
        mode='val'
    )
    
    # Отримання передбачень
    predictions = []
    ground_truths = []
    
    for i in range(len(val_dataset)):
        image, target = val_dataset[i]
        pred = model.predict([image], threshold=args.threshold)[0]
        
        predictions.append(pred)
        ground_truths.append(target)
    
    # Обчислення метрик
    metrics_calculator = DetectionMetrics(num_classes=args.num_classes)
    results = metrics_calculator.evaluate_model(predictions, ground_truths)
    
    # Виведення результатів
    print("\nРЕЗУЛЬТАТИ ОЦІНКИ:")
    print("-" * 60)
    print("\nЗагальні метрики:")
    for key, value in results['overall'].items():
        print(f"  {key}: {value:.4f}")
    
    # Генерація звіту
    report_gen = ReportGenerator()
    report_gen.add_evaluation_metrics(results)
    report_gen.generate_full_report()
    
    print("\n✓ Оцінка завершена!")


def demo_mode():
    """Демонстраційний режим з прикладом використання"""
    print(f"\n{'='*60}")
    print(f"ДЕМОНСТРАЦІЙНИЙ РЕЖИМ")
    print(f"{'='*60}\n")
    
    # Створення структури директорій
    create_directory_structure()
    
    # Створення тестового датасету
    demo_dir = os.path.join(Config.DATA_DIR, 'demo')
    if not os.path.exists(demo_dir):
        print("Створення демонстраційного датасету...")
        DatasetBuilder.create_sample_dataset(demo_dir, num_samples=5)
    
    # Ініціалізація моделі
    model = FasterRCNNDetector(num_classes=Config.NUM_CLASSES, pretrained=True)
    visualizer = DetectionVisualizer()
    
    # Детекція на всіх зображеннях
    report_gen = ReportGenerator()
    report_gen.add_model_info({
        'model_name': 'Faster R-CNN ResNet50-FPN',
        'num_classes': Config.NUM_CLASSES,
        'device': str(Config.DEVICE),
        'detection_threshold': Config.DETECTION_THRESHOLD
    })
    
    image_files = [f for f in os.listdir(demo_dir) if f.endswith('.jpg')]
    all_predictions = []
    
    for img_file in image_files[:3]:  # Обробка перших 3 зображень
        img_path = os.path.join(demo_dir, img_file)
        predictions = detect_objects(img_path, model, visualizer, Config.DETECTION_THRESHOLD)
        
        pred = predictions[0]
        classes = [Config.COCO_CLASSES[l] for l in pred['labels'].cpu().numpy() 
                  if l < len(Config.COCO_CLASSES)]
        
        report_gen.add_detection_result(img_file, len(pred['boxes']), classes)
        all_predictions.append(pred)
    
    # Статистика
    if all_predictions:
        stats_path = os.path.join(Config.OUTPUT_DIR, 'detection_statistics.png')
        visualizer.plot_detection_statistics(all_predictions, save_path=stats_path)
    
    # Генерація повного звіту
    report_gen.generate_full_report()
    
    print("\n✓ Демонстрацію завершено!")
    print(f"  Результати збережено в: {Config.OUTPUT_DIR}")
    print(f"  Звіти збережено в: {Config.REPORTS_DIR}")


def main():
    """Головна функція програми"""
    # Парсинг аргументів
    args = parse_arguments()
    
    print("\n" + "="*70)
    print(" "*15 + "СИСТЕМА РОЗПІЗНАВАННЯ ОБ'ЄКТІВ")
    print(" "*20 + "Faster R-CNN ResNet50-FPN")
    print("="*70)
    
    # Створення структури директорій
    create_directory_structure()
    
    # Виконання відповідного режиму
    if args.mode == 'detect':
        if args.image is None:
            print("\n❌ Помилка: необхідно вказати шлях до зображення (--image)")
            return
        
        model = FasterRCNNDetector(num_classes=args.num_classes, pretrained=True)
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_checkpoint(args.checkpoint)
        
        visualizer = DetectionVisualizer()
        detect_objects(args.image, model, visualizer, args.threshold)
    
    elif args.mode == 'train':
        train_model(args)
    
    elif args.mode == 'evaluate':
        evaluate_model(args)
    
    elif args.mode == 'demo':
        demo_mode()
    
    print("\n" + "="*70)
    print("ПРОГРАМА ЗАВЕРШЕНА")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
