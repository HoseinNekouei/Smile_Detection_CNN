import numpy as np
import cv2 as cv
from mtcnn import MTCNN
from keras.models import load_model

detector = MTCNN()
REDCOLOR = (0, 0, 255)
GREENCOLOR = (0, 255, 0)
model = load_model(r'deep_learning\Week6\smile_detector\smile_model.h5')

cap = cv.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    out = detector.detect_faces(frame_rgb)

    if len(out) > 0:
        for face in out:
            x, y, w, h = face['box']
            face = frame[y:y+h, x:x+w]

            face = cv.resize(face, (32, 32))
            face = face / 255

            face = np.array([face])

            predict = model.predict(face)[0][0]
            predict = np.round(predict)

            if predict == 1.0:
                COLOR = GREENCOLOR
                label = 'smile'
            else:
                COLOR = REDCOLOR
                label = 'no_smile'

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
