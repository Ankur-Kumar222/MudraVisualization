import os
import cv2

input_directory = 'Captured_Hands_Cropped_More'

for folder in os.listdir(input_directory):
    path = os.path.join(input_directory, folder)

    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)

                flipped_image = cv2.flip(image, 1)  #1 means horizontal flip so left becomes right and vice versa 

                output_path = os.path.join(path, f'flipped_{filename}') #same name in same folder but with flipped written in front

                cv2.imwrite(output_path, flipped_image)