# MudraVisualization

Group Project by: Ankur Kumar and Mahika Nair

Indian classical dance mudra (hand gesture) recognition using YOLOv8 image classification and MediaPipe hand detection. Classifies 30 Asamyukta (single-hand) mudras in real-time video.

## Quick Start (Real-Time Detection)

Requires Python 3.12 (MediaPipe does not yet support 3.13+).

```bash
# Create a virtual environment with uv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install opencv-python mediapipe ultralytics

# Run the live demo
python mudra_live.py
```

A webcam window will open. Hold up a hand gesture to see predictions. Press **Esc** to quit.

> **macOS note:** If no window appears, check your Dock for the Python icon and click it. Your terminal app also needs Camera permission under System Settings > Privacy & Security > Camera.

Alternatively, you can run the notebook version:

```bash
uv pip install jupyter
jupyter notebook VideoYOLO.ipynb
```

## Full Setup (Training from Scratch)

```bash
uv pip install opencv-python mediapipe ultralytics scikit-learn matplotlib numpy pillow tensorflow
```

## Project Structure

```
mudra_live.py            # Standalone real-time mudra recognition (recommended)
VideoYOLO.ipynb          # Notebook version of real-time recognition
collect_imgs.py          # Step 1: Capture hand images from webcam (press P to start each class)
Image_cropper.py         # Step 2: Crop hands using MediaPipe landmark detection
Image_Flipper.py         # Step 3: Augment dataset with horizontal flips
YOLO_Model.ipynb         # Step 4: Train YOLOv8 classifier (yolov8n-cls)
Metrics_yolo.ipynb       # Step 5: Evaluate model (accuracy, precision, recall, confusion matrix)
runs/classify/train3/weights/best.pt  # Trained model weights (required for inference)
```

## Pipeline Details

1. **Data collection** (`collect_imgs.py`): Captures webcam frames into `Captured_Hands/` with 30 class subdirectories. Default: 20 images per class. Press P to start each class.
2. **Cropping** (`Image_cropper.py`): Uses MediaPipe to detect hand landmarks, crops a square region around the hand. Input: `Captured_Hands_More/`, Output: `Captured_Hands_Cropped_More/`.
3. **Augmentation** (`Image_Flipper.py`): Horizontally flips all cropped images, saving with `flipped_` prefix in the same directory. Doubles dataset size.
4. **Training** (`YOLO_Model.ipynb`): Fine-tunes `yolov8n-cls.pt` (pretrained on ImageNet) on the mudra dataset. Trains at 64x64 resolution.
5. **Evaluation** (`Metrics_yolo.ipynb`): Computes accuracy (achieved 80%), per-class precision/recall/F1, and confusion matrix using scikit-learn.
6. **Inference** (`mudra_live.py` / `VideoYOLO.ipynb`): MediaPipe detects hands in webcam frames, crops ROI, YOLO classifies the mudra. Supports 1-2 simultaneous hands.

## Configuration

Both `mudra_live.py` and `VideoYOLO.ipynb` have a `WEBCAM_INDEX` variable at the top. Change it to `1` if you have an external webcam.

If you retrain the model, update the model path and regenerate the `labels_dict` mapping to match your new training run's class ordering (check `runs/classify/trainN/`).

## 30 Mudra Classes (Asamyukta)

1. Pataka
2. Tripataka
3. Ardhapataka
4. Kartarimukha
5. Mayura
6. Ardhachandra
7. Arala
8. Shukatunda
9. Mushthi
10. Shikhara
11. Kapitha
12. Katakamukha-1
13. Katakamukha-2
14. Katakamukha-3
15. Suchi
16. Chandrakala
17. Padmakosha
18. Sarpashisha
19. Mrighashisha
20. Singhamukha
21. Kangula
22. Alapadma
23. Chatura
24. Bhrahmara
25. Hamsasya
26. Hamsapaksha
27. Santamsha
28. Mukula
29. Tamrachuda
30. Trishula

## Troubleshooting

- **No webcam window appears (macOS):** Check your Dock for the Python icon. Ensure your terminal has Camera permission in System Settings.
- **`module 'mediapipe' has no attribute 'solutions'`:** You need Python 3.12. MediaPipe 0.10.21+ (required for Python 3.13) removed the legacy `mp.solutions` API this project uses.
- **No webcam detected:** Try changing `WEBCAM_INDEX` to `0`, `1`, or `2`.
- **MediaPipe not detecting hands:** Ensure good lighting and hands are clearly visible.
- **Wrong predictions after retraining:** The `labels_dict` mapping must match your new training run's class ordering.
