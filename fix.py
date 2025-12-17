import json
import math

INPUT_JSON = r"C:\Users\38066\Desktop\photo\val\annotations.json"
OUTPUT_JSON = r"C:\Users\38066\Desktop\photo\val\annotations_fixed.json"

MIN_SIZE = 2.0   # мінімальна ширина / висота box-а (у пікселях)

def is_finite_box(box):
    return all(math.isfinite(v) for v in box)

def fix_box(box):
    x1, y1, x2, y2 = box

    # міняємо місцями якщо переплутано
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    w = x2 - x1
    h = y2 - y1

    if w < MIN_SIZE or h < MIN_SIZE:
        return None

    return [float(x1), float(y1), float(x2), float(y2)]

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

total_boxes = 0
removed_boxes = 0
fixed_boxes = 0
removed_images = 0

clean_data = []

for item in data:
    new_boxes = []
    new_labels = []

    for box, label in zip(item["boxes"], item["labels"]):
        total_boxes += 1

        if not is_finite_box(box):
            removed_boxes += 1
            continue

        fixed = fix_box(box)
        if fixed is None:
            removed_boxes += 1
            continue

        if fixed != box:
            fixed_boxes += 1

        new_boxes.append(fixed)
        new_labels.append(int(label))

    # ❗ Faster R-CNN НЕ любить картинки без box-ів
    if len(new_boxes) == 0:
        removed_images += 1
        continue

    item["boxes"] = new_boxes
    item["labels"] = new_labels
    clean_data.append(item)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=2)

print("===== CLEANING REPORT =====")
print(f"Всього box-ів: {total_boxes}")
print(f"Виправлено box-ів: {fixed_boxes}")
print(f"Видалено box-ів: {removed_boxes}")
print(f"Видалено зображень без box-ів: {removed_images}")
print(f"✔ Очищений файл збережено як: {OUTPUT_JSON}")
