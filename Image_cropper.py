import os
import cv2
import mediapipe as mp 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #as defined in notebook

input_directory = 'Captured_Hands_More'
output_directory = 'Captured_Hands_Cropped_More'

def create_output_directory(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for dir_name in dirs:
            output_subdir = os.path.join(output_path, os.path.relpath(os.path.join(root, dir_name), input_path))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

create_output_directory(input_directory, output_directory)


for folder in os.listdir(input_directory):
    path = os.path.join(input_directory, folder)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                #taken from our VideoYOLO notebook
                x_land = []
                y_land = []
                frame = cv2.imread(file_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

                h, w, _ = frame.shape
                results = hands.process(frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_land.append(x)
                            y_land.append(y)

                    x1 = int(min(x_land) * w)
                    y1 = int(min(y_land) * h)

                    x2 = int(max(x_land) * w)
                    y2 = int(max(y_land) * h)
                    width = x2 - x1
                    height = y2 - y1

                    max_side = max(width, height)
                    new_x1 = x1 + (width - max_side) // 2 - 20
                    new_y1 = y1 + (height - max_side) // 2 - 20
                    new_x2 = new_x1 + max_side + 20
                    new_y2 = new_y1 + max_side + 20

                    new_x1 = abs(new_x1)
                    new_x2 = abs(new_x2)
                    new_y1 = abs(new_y1)
                    new_y2 = abs(new_y2)

                    cropped_frame = frame[new_y1:new_y2, new_x1:new_x2] #this cropped frame becomes our final image for training/testing

                output_subdir = os.path.join(output_directory, folder)
                output_filename = f'{filename}'  
                output_path = os.path.join(output_subdir, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))