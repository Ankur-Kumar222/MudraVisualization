import os
import cv2

directory = 'Captured_Hands'
if not os.path.exists(directory ):
    os.makedirs(directory)

n = 30 #no.of classes
dataset_size = 20 #no. of images per class 

cap = cv2.VideoCapture(0)
for j in range(n):
    if not os.path.exists(os.path.join(directory , str(j))):
        os.makedirs(os.path.join(directory , str(j))) #making the respective directories 

    print(f'Collecting data for class {j}')

    done = False
    while True: #we wait to press a button to start taking the pictures
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "P" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('p'):
            break

    counter = 0
    while counter < dataset_size: #start taking pictures 
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(directory , str(j), 'SUBJ6{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
