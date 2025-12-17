import os
import json
from PIL import Image

IMG_DIR = r"E:\data\val\images"
LBL_DIR = r"E:\data\val\labels"
ANN_FILE = r"E:\data\val\annotations\annotations.json"

# YOLO class_id -> твій label (ВАЖЛИВО: Faster R-CNN не любить 0)
CLASS_MAP = {
    0: 1,  # person
    1: 2,  # car
    2: 3,  # dog
    3: 4   # chair
}

os.makedirs(os.path.dirname(ANN_FILE), exist_ok=True)

annotations = []

def fix_zero(v):
    """Не дозволяє координатам бути 0 або менше"""
    return 1 if v <= 0 else v

for file in sorted(os.listdir(LBL_DIR)):
    if not file.endswith(".txt"):
        continue

    image_name = file.replace(".txt", ".jpg")
    image_path = os.path.join(IMG_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"⚠ Пропущено (нема зображення): {image_name}")
        continue

    img = Image.open(image_path)
    W, H = img.size

    boxes = []
    labels = []

    with open(os.path.join(LBL_DIR, file), "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())

            # YOLO → Faster R-CNN формат
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H

            # 🔒 захист від 0
            x1 = fix_zero(x1)
            y1 = fix_zero(y1)
            x2 = fix_zero(x2)
            y2 = fix_zero(y2)

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_MAP[int(cls)])

    # ❗ якщо на фото нема об'єктів — пропускаємо
    if len(boxes) == 0:
        continue

    annotations.append({
        "image_path": image_name,  # ТІЛЬКИ імʼя файлу
        "boxes": boxes,
        "labels": labels
    })

# Збереження одного annotations.json
with open(ANN_FILE, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"✅ Готово! Створено {len(annotations)} анотацій")
print(f"📄 Файл: {ANN_FILE}")
