# YOLOv6 Object Detection - Fine-tuning on Custom Dataset

Fine-tuning YOLOv6-n (nano) model for object detection on custom datasets. This project demonstrates transfer learning by adapting a pretrained YOLO model to detect objects in specialized domains (pets, birds, or dogs).

##  Project Overview

This project implements object detection using the YOLOv6-n architecture, focusing on:
- Testing pretrained YOLOv6-n on a target dataset
- Fine-tuning the model on domain-specific data
- Evaluating performance improvements through transfer learning

**Course**: EECE 7398 - Advances in Deep Learning  
**Institution**: Northeastern University  
**Task**: Lab 3 - Object Detection

##  Datasets

Choose one of the following datasets for fine-tuning:

| Dataset | Images | Classes | Use Case |
|---------|--------|---------|----------|
| [Oxford-IIIT Pet](http://www.robots.ox.ac.uk/~vgg/data/pets/) | ~7,000 | 37 breeds | Pet detection |


##  Model Architecture

**YOLOv6-n (Nano)**
- Lightweight version of YOLOv6 optimized for speed
- Single-stage object detector
- Anchor-free design
- Suitable for edge devices and real-time applications

### Key Features:
- **Backbone**: EfficientRep
- **Neck**: Rep-PAN
- **Head**: Efficient decoupled head
- **Parameters**: ~4.7M
- **Speed**: Fast inference for real-time detection

```

##  Usage

### 1. Testing Pretrained Model (Before Fine-tuning)

Test the pretrained model on your target dataset:
```bash
python tools/eval.py \
    --data data/custom.yaml \
    --weights weights/yolov6n.pt \
    --batch-size 32 \
    --img-size 640 \
    --task val \
    --device 0
```

**Expected Output:**
```
Average Precision (AP) @ IoU=0.50:0.95 | area=all | maxDets=100 = 0.313
Average Precision (AP) @ IoU=0.50      | area=all | maxDets=100 = 0.460
Average Precision (AP) @ IoU=0.75      | area=all | maxDets=100 = 0.332
...
```

### 2. Fine-tuning on Target Dataset

Fine-tune the pretrained model:
```bash
python tools/train.py \
    --batch 32 \
    --conf configs/yolov6n_finetune.py \
    --data data/custom.yaml \
    --epochs 100 \
    --device 0 \
    --img-size 640 \
    --name yolov6n_custom
```

**Training Arguments:**
- `--batch`: Batch size (adjust based on GPU memory)
- `--epochs`: Number of training epochs
- `--img-size`: Input image size
- `--device`: GPU device ID (use 'cpu' for CPU training)
- `--workers`: Number of data loading workers

**Training Progress:**
```
Epoch   GPU_mem   box_loss   obj_loss   cls_loss   Instances   Size
1/100   4.8G      0.0823     0.0421     0.0193     128         640
2/100   4.8G      0.0735     0.0389     0.0165     128         640
...
```

### 3. Testing Fine-tuned Model (After Fine-tuning)

Evaluate the fine-tuned model:
```bash
python tools/eval.py \
    --data data/custom.yaml \
    --weights runs/train/yolov6n_custom/weights/best_ckpt.pt \
    --batch-size 32 \
    --img-size 640 \
    --task val \
    --device 0
```

### 4. Inference on Test Images

Run detection on sample images:
```bash
python tools/infer.py \
    --weights runs/train/yolov6n_custom/weights/best_ckpt.pt \
    --source ./test_images/ \
    --img-size 640 \
    --device 0
```


##  Configuration

### Fine-tuning Configuration (`configs/yolov6n_finetune.py`)
```python
model = dict(
    type='YOLOv6n',
    pretrained='weights/yolov6n.pt',
    depth_multiple=0.33,
    width_multiple=0.25,
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.01,              # Initial learning rate
    lrf=0.01,              # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
```

##  Training Tips

### For Better Performance

1. **Adjust Learning Rate**
```python
   lr0=0.001  # Lower for fine-tuning
```

2. **Increase Training Epochs**
```bash
   --epochs 150  # More epochs for convergence
```

3. **Use Data Augmentation**
   - Mosaic augmentation
   - HSV color space augmentation
   - Random affine transformations

4. **Freeze Early Layers** (optional)
```python
   freeze = [0, 1, 2, 3]  # Freeze backbone layers
```

5. **Multi-scale Training**
```bash
   --img-size 640 --multi-scale
```


## 🎓 Key Concepts

### Transfer Learning
Fine-tuning leverages knowledge learned from COCO dataset:
- **Frozen layers**: Low-level feature detectors (edges, textures)
- **Trainable layers**: High-level feature detectors (domain-specific patterns)

### Evaluation Metrics

**mAP (mean Average Precision)**
- Primary metric for object detection
- Averaged precision across IoU thresholds (0.5 to 0.95)

**Precision**
- TP / (TP + FP)
- Percentage of correct positive predictions

**Recall**
- TP / (TP + FN)
- Percentage of actual positives detected

