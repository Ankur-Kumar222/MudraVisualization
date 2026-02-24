# MudraVisualization

Indian classical dance mudra (hand gesture) recognition using YOLOv8 image classification and MediaPipe hand detection. Classifies 30 Asamyukta (single-hand) mudras in real-time video.

## Quick Start (Real-Time Detection Only)

```bash
pip install opencv-python mediapipe ultralytics
```

Open `VideoYOLO.ipynb` and run all cells. Requires webcam access. Press Esc to stop.

## Full Setup (Training from Scratch)

```bash
pip install opencv-python mediapipe ultralytics scikit-learn matplotlib numpy pillow
```

## Project Structure

```
collect_imgs.py          # Step 1: Capture hand images from webcam (press P to start each class)
Image_cropper.py         # Step 2: Crop hands using MediaPipe landmark detection
Image_Flipper.py         # Step 3: Augment dataset with horizontal flips
YOLO_Model.ipynb         # Step 4: Train YOLOv8 classifier (yolov8n-cls)
Metrics_yolo.ipynb       # Step 5: Evaluate model (accuracy, precision, recall, confusion matrix)
VideoYOLO.ipynb          # Step 6: Real-time mudra recognition from webcam
runs/classify/train3/weights/best.pt  # Trained model weights (required for inference)
```

## Pipeline Details

1. **Data collection** (`collect_imgs.py`): Captures webcam frames into `Captured_Hands/` with 30 class subdirectories. Default: 20 images per class. Press P to start each class.
2. **Cropping** (`Image_cropper.py`): Uses MediaPipe to detect hand landmarks, crops a square region around the hand. Input: `Captured_Hands_More/`, Output: `Captured_Hands_Cropped_More/`.
3. **Augmentation** (`Image_Flipper.py`): Horizontally flips all cropped images, saving with `flipped_` prefix in the same directory. Doubles dataset size.
4. **Training** (`YOLO_Model.ipynb`): Fine-tunes `yolov8n-cls.pt` (pretrained on ImageNet) on the mudra dataset. Trains at 64x64 resolution.
5. **Evaluation** (`Metrics_yolo.ipynb`): Computes accuracy (achieved 80%), per-class precision/recall/F1, and confusion matrix using scikit-learn.
6. **Inference** (`VideoYOLO.ipynb`): MediaPipe detects hands in webcam frames, crops ROI, YOLO classifies the mudra. Supports 1-2 simultaneous hands.

## Hardcoded Paths to Update

The notebooks contain hardcoded paths from the original authors. Update these before running:

- `YOLO_Model.ipynb` cell 2: `data=` path to your dataset directory
- `VideoYOLO.ipynb` cell 1: model path → use `runs/classify/train3/weights/best.pt`
- `VideoYOLO.ipynb` cell 3: `cv2.VideoCapture(1)` → change to `0` if you have a single webcam
- `Metrics_yolo.ipynb` cell 0: model path → use `runs/classify/train3/weights/best.pt`
- `Metrics_yolo.ipynb` cell 1: `directory` → path to your test dataset

## Key Technical Details

- **Model**: YOLOv8 nano classification (`yolov8n-cls`), fine-tuned for 10 epochs at 64x64
- **Hand detection**: MediaPipe Hands with `static_image_mode=True`, `min_detection_confidence=0.3`
- **Cropping**: Square bounding box around hand landmarks with 20px padding on each side
- **Label mapping**: YOLO's internal class ordering differs from the folder numbering. `labels_dict` in both `VideoYOLO.ipynb` and `Metrics_yolo.ipynb` handles this remapping. If you retrain, you must regenerate this mapping.

## 30 Mudra Classes (Asamyukta)

Pataka, Tripataka, Ardhapataka, Kartarimukha, Mayura, Ardhachandra, Arala, Shukatunda, Mushthi, Shikhara, Kapitha, Katakamukha (1-3), Suchi, Chandrakala, Padmakosha, Sarpashisha, Mrighashisha, Singhamukha, Kangula, Alapadma, Chatura, Bhrahmara, Hamsasya, Hamsapaksha, Santamsha, Mukula, Tamrachuda, Trishula

## Common Issues

- **No webcam detected**: Check `cv2.VideoCapture()` index (try 0, 1, or 2)
- **MediaPipe not detecting hands**: Ensure good lighting and hands are clearly visible
- **Wrong predictions after retraining**: The `labels_dict` mapping must match your new training run's class ordering (check `runs/classify/trainN/` for the mapping)
- **Import errors**: Make sure all dependencies are installed with the correct Python version (3.8+)
