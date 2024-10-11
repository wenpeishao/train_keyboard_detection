import os
import json
import shutil
import glob
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Set up directories and data paths
data_dir = os.getcwd()
print(f"Current working directory: {data_dir}")

# Paths to annotation files (assumed to be already extracted by the shell script)
train_annotations_path = 'annotations/annotations/instances_train2017.json'
val_annotations_path = 'annotations/annotations/instances_val2017.json'

# Load annotations
with open(train_annotations_path, 'r') as f:
    train_annotations = json.load(f)
with open(val_annotations_path, 'r') as f:
    val_annotations = json.load(f)

# Get the category ID for 'keyboard'
keyboard_category_id = next(
    (category['id'] for category in train_annotations['categories'] if category['name'] == 'keyboard'),
    None
)
print(f"Keyboard category ID: {keyboard_category_id}")

if keyboard_category_id is None:
    raise ValueError("Keyboard category ID not found in the annotations.")

# Helper function to get image IDs for the keyboard category
def get_keyboard_image_ids(annotations, category_id):
    return {annotation['image_id'] for annotation in annotations['annotations'] if annotation['category_id'] == category_id}

train_keyboard_image_ids = get_keyboard_image_ids(train_annotations, keyboard_category_id)
val_keyboard_image_ids = get_keyboard_image_ids(val_annotations, keyboard_category_id)

print(f"Number of training images with keyboards: {len(train_keyboard_image_ids)}")
print(f"Number of validation images with keyboards: {len(val_keyboard_image_ids)}")

# Helper function to get image file names
def get_image_file_names(annotations, image_ids):
    return [image['file_name'] for image in annotations['images'] if image['id'] in image_ids]

train_keyboard_images = get_image_file_names(train_annotations, train_keyboard_image_ids)
val_keyboard_images = get_image_file_names(val_annotations, val_keyboard_image_ids)

# Create directories for the keyboard dataset
os.makedirs('keyboard_dataset/train/images', exist_ok=True)
os.makedirs('keyboard_dataset/train/labels', exist_ok=True)
os.makedirs('keyboard_dataset/val/images', exist_ok=True)
os.makedirs('keyboard_dataset/val/labels', exist_ok=True)

# Copy training and validation images to the corresponding directories
def copy_images(image_list, src_dir, dst_dir):
    for file_name in image_list:
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"File {src} not found. Skipping...")

# Use absolute paths for source directories
train_image_src_dir = os.path.abspath('train2017/train2017')
val_image_src_dir = os.path.abspath('val2017/val2017')

print(f"Copying training images from {train_image_src_dir} to keyboard_dataset/train/images...")
copy_images(train_keyboard_images, train_image_src_dir, 'keyboard_dataset/train/images')

print(f"Copying validation images from {val_image_src_dir} to keyboard_dataset/val/images...")
copy_images(val_keyboard_images, val_image_src_dir, 'keyboard_dataset/val/images')

# Function to convert annotations to YOLO format
def convert_annotations(annotations, image_ids, output_dir, category_id):
    annotations_index = {}
    for ann in annotations['annotations']:
        if ann['image_id'] in image_ids and ann['category_id'] == category_id:
            annotations_index.setdefault(ann['image_id'], []).append(ann)

    images_index = {img['id']: img for img in annotations['images'] if img['id'] in image_ids}

    for image_id, anns in annotations_index.items():
        img = images_index[image_id]
        img_width = img['width']
        img_height = img['height']
        label_file_name = os.path.splitext(img['file_name'])[0] + '.txt'
        label_file_path = os.path.join(output_dir, label_file_name)

        with open(label_file_path, 'w') as f:
            for ann in anns:
                bbox = ann['bbox']
                # Convert COCO bbox format to YOLO format
                x_min = bbox[0]
                y_min = bbox[1]
                width = bbox[2]
                height = bbox[3]

                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                w = width / img_width
                h = height / img_height

                class_id = 0  # 'keyboard' is the only class

                f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# Convert annotations for training and validation images
convert_annotations(train_annotations, train_keyboard_image_ids, 'keyboard_dataset/train/labels', keyboard_category_id)
convert_annotations(val_annotations, val_keyboard_image_ids, 'keyboard_dataset/val/labels', keyboard_category_id)

# Save the configuration file for YOLO with absolute paths
data_config = {
    'path': os.path.abspath('./keyboard_dataset'),
    'train': os.path.abspath('keyboard_dataset/train/images'),
    'val': os.path.abspath('keyboard_dataset/val/images'),
    'names': ['keyboard'],
    'nc': 1
}

save_path = './keyboard_dataset/data_config.yaml'

# After copying the images, list the contents of the directories
print("Listing training image directory contents:")
print(os.listdir('keyboard_dataset/train/images'))

print("Listing validation image directory contents:")
print(os.listdir('keyboard_dataset/val/images'))

# Save the YOLO configuration
with open(save_path, 'w') as f:
    yaml.dump(data_config, f)

print(f"Dataset configuration saved to {save_path}")

# Load and train YOLOv8 model with compatible arguments
model = YOLO('yolo11s.pt')  # Ensure the model path is correct and the pre-trained weights are available
print(f"Training YOLO model using dataset configuration: {save_path}")

# Adjusted training settings to include only supported arguments
model.train(
    data=save_path,
    epochs=100,
    imgsz=416,  # Reduce image size for memory efficiency
    batch=8,  # Lower batch size to avoid memory issues
    name='keyboard_detection',
    lr0=0.001,  # Set initial learning rate
    lrf=0.01,  # Final learning rate after warm-up
    weight_decay=0.0001,  # Regularization
    momentum=0.85,  # Momentum for optimization
    mosaic=0.5,  # Use mosaic data augmentation
    mixup=0.5,  # Use mixup data augmentation
    augment=True,  # General data augmentation
    patience=20,  # Early stopping
    optimizer='Adam'  # Use Adam optimizer
)

# Evaluate the model and get metrics
metrics = model.val(data=save_path, conf=0.25, iou=0.65)  # Adjust IoU and confidence thresholds
print(f"Model evaluation completed. Validation metrics: {metrics}")

# Perform inference and calculate ROC AUC
print("Calculating ROC AUC and other performance metrics...")

val_image_paths = glob.glob('keyboard_dataset/val/images/*.jpg')
all_scores = []
all_labels = []
iou_threshold = 0.5

# Function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Process each validation image
for img_path in tqdm(val_image_paths):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # Load ground truth boxes
    label_path = os.path.join('keyboard_dataset/val/labels', img_id + '.txt')
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            gt_boxes.append([x_min, y_min, x_max, y_max])

    # Get predictions
    result = model(img_path)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    # Determine TPs and FPs
    matched_gt = set()
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        score = scores[i]
        max_iou = 0
        max_iou_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = j
        if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
            # True Positive
            all_scores.append(score)
            all_labels.append(1)
            matched_gt.add(max_iou_idx)
        else:
            # False Positive
            all_scores.append(score)
            all_labels.append(0)

# Calculate ROC AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = roc_auc_score(all_labels, all_scores)

print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Keyboard Detection')
plt.legend(loc="lower right")
plt.show()
