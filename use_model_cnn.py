import numpy as np
import cv2 as cv
import uuid
# from mtcnn import MTCNN
from keras.models import load_model

# detector = MTCNN()
   # Loading the required haar-cascade xml classifier file
REDCOLOR = (0, 0, 255)
GREENCOLOR = (0, 255, 0)
model = load_model(r'smile_model.h5')
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(frame, 1.1, 9)

    # out = detector.detect_faces(frame_rgb)

    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face = rgb_frame[y:y+h, x:x+w]

            face = cv.resize(face, (64, 64))
            face = face / 255
            face = np.array([face])
            # print(face)

            base_predict = model.predict(face)[0][0]
            predict = np.round(base_predict)

            if predict == 1.0:
                COLOR = GREENCOLOR
                label = f'smile : {base_predict:.2f}'
                fileName = uuid.uuid1().hex
                cv.imwrite(f'photo/{fileName}.jpg', frame)

            else:
                COLOR = REDCOLOR
                label = f'no smile : {base_predict:.2f}'

            cv.rectangle(frame, (x, y), (x+w, y+h), COLOR, 2)

            font = cv.FONT_HERSHEY_TRIPLEX
            # get text size
            (text_width, text_heigh), baseline = cv.getTextSize(label, font, 1, 2)
            # draw a rectangle above object rectangle
            cv.rectangle(frame, (x, y), (x-5+text_width, y -
                         text_heigh-baseline), COLOR, thickness=-1)

            cv.putText(frame,label, (x, y-6),
                       font, 1, (0, 0, 0), 1)

        else:
            print('No face detected')

    cv.imshow('Smile Detection', frame)
    if cv.waitKey(19) == ord('q'):
        break


cv.destroyAllWindows()
cap.release()
