# Training Report
**Author:** Mayank Dhapodkar

## Dataset Strategy
Three datasets combined on Roboflow:
- Aerial Garbage Detection (Roboflow)
- Kaggle Drone Garbage (~1500 images, no annotations — annotated manually using Roboflow AI assist)
- TACO (outdoor litter, real-world scenes)

All uploaded individually to Roboflow, merged into single project, exported in YOLOv8 format.
Single class: `Garbage`

| Split | Images |
|-------|--------|
| Train | 1716 |
| Val | 455 |
| Test | 234 |
| Total | 2405 |

## Model
- Architecture: YOLOv8n
- Pretrained weights: COCO (yolov8n.pt)
- Fine-tuned — NOT trained from scratch
- Platform: Google Colab T4 GPU

## Augmentations (applied by Roboflow + YOLOv8)
| Augmentation | Value | Why |
|---|---|---|
| Horizontal flip | 0.5 | Garbage appears in any orientation |
| Mosaic | 1.0 | Combines 4 images, improves small object detection |
| HSV Hue | 0.015 | Handles lighting variation |
| HSV Saturation | 0.7 | Different environments, times of day |
| HSV Value | 0.4 | Brightness/contrast variation |
| Scale jitter | 0.5 | Detects garbage at various distances |
| Random erasing | 0.4 | Handles partial occlusion |
| Auto augment | randaugment | Additional robustness |

## Training Config
- Epochs: 75 (patience=100, ran full)
- Batch size: 16
- Image size: 640x640
- Optimizer: Auto (AdamW)
- Learning rate: 0.01

## Results

### Validation Set
| Metric | Score |
|--------|-------|
| mAP@50 | 84.1% |
| mAP@50-95 | 39.0% |

### Test Set (unseen data)
| Metric | Score |
|--------|-------|
| mAP@50 | 81.2% |
| mAP@50-95 | 37.5% |

### Per-class breakdown
| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Garbage | 0.84 | 0.781 | 81.2% |

## Why stopped at 75 epochs
Ran full 75 epochs. Validation mAP converged — no significant improvement observed after epoch ~60. Early stopping patience was 100 so ran to completion.

## Artifacts
- `results.png` — loss and mAP curves across all epochs
- `confusion_matrix.png` — per-class prediction breakdown on validation set
- `val_predictions.jpg` — sample predictions on unseen validation images
<<<<<<< HEAD
- `best.pt` — best model weights saved at highest validation mAP
=======
- `best.pt` — best model weights saved at highest validation mAP
>>>>>>> c7dc184 (Added fastAPI /predict endpoint)
