# MudraVisualisation

This project focuses on building a hand gesture recognition system, specifically targeting Indian classical dance mudras. The notebooks and scripts provided in this repository guide you through the process of collecting, preprocessing, training, and evaluating a model for recognizing these intricate hand movements in real-time video.

There are 30 Indian Classical Mudras being classified here (the Asamyukta mudras):
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


## collect_imgs.py (credits: https://www.youtube.com/watch?v=MJCSjXepaAM)

This script captures images from a live video feed using OpenCV. It allows the user to collect a dataset of hand images for various classes. The number of classes (n), images per class (dataset_size), and other parameters can be adjusted.

## image_cropper.py

This script uses the MediaPipe library to crop hand images captured by the collect_imgs.py script. It detects hand landmarks and crops the images around the detected hand region. Cropped images are then saved to a specified output directory.

## image_flipper.py

This script takes the cropped hand images from the image_cropper.py script and creates horizontally flipped versions. The flipped images are saved with a 'flipped_' prefix in the same directory.

## YOLO_model.ipynb (credit: https://docs.ultralytics.com/tasks/classify/#models)

This Jupyter notebook demonstrates the usage of the Ultralytics YOLO (You Only Look Once) framework for training a model on hand images. It uses a pre-trained YOLO model and fine-tunes it on a specified dataset, showing how to train the model for a certain number of epochs and image size.

## Metrics_YOLO.ipynb

This Jupyter notebook evaluates the trained YOLO model from YOLO_model.ipynb. It calculates and displays accuracy, precision, recall, F1-score, and confusion matrix using scikit-learn metrics from the test set. It also maps the predicted class labels to the original class labels based on a provided mapping.

## VideoYOLO.ipynb

This Jupyter notebook integrates hand detection using the MediaPipe library with a pre-trained YOLO model for mudra (hand gesture) recognition. It captures live video, detects hands, crops the region of interest, and predicts the mudra using the YOLO model. Predictions are displayed in real-time along with bounding boxes on the detected hands. Here are the two essential functions:

### predict_from_video(img)
This function takes a frame from the video feed, saves it as a temporary image file, and utilizes the pre-trained YOLO model to predict the class (mudra) of the hand in the image. The predicted class is returned.

### box_maker(mhl, h, w)
This function processes the detected hand landmarks (mhl) using the MediaPipe library. It calculates the bounding box coordinates for the hands in the frame. The coordinates are normalized and then converted to the absolute dimensions of the frame. The function returns the coordinates for both the rectangular box and the square box around the Region of Interest (ROI).

## Note: To run the video detection directly, all you need is the VideoYOLO.ipynb and the '/runs/classify/train3/weights/best.pt' file.
