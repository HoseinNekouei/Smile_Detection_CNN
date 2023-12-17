import numpy as np
import cv2 as cv
from glob import glob
from keras.models import load_model

REDCOLOR = (0, 0, 255)
GREENCOLOR = (0, 255, 0)
labels = ['not smile', 'smile']
model = load_model(r'smile_model.h5')
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

for item in glob(r'Smile_sample\*.jpg'):

    img = cv.imread(item)

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(img, 1.1, 9)

    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:
        
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            face = img_rgb[y:y+h, x:x+w]

            face = cv.resize(face, (64, 64))
            face = face / 255
            # print(face.shape)
            normalized_face = np.array([face])   
            # print(face.shape)

            out = model.predict(normalized_face)[0][0]

            max_index = int(np.round(out))
            predict = labels[max_index]
            probability = out

            color  = REDCOLOR if max_index == 0 else GREENCOLOR
            text = f'{predict} : {probability:.2f}'
            font = cv.FONT_HERSHEY_TRIPLEX

            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)

            # get text size
            (text_width, text_heigh), baseline = cv.getTextSize(text, font, 1, 2)
            
            # draw a rectangle above object rectangle
            cv.rectangle(img, (x, y), (x-5+text_width, y - text_heigh-baseline), color, thickness=-1)

            cv.putText(img, text, (x, y-6), font, 0.6, (0, 0, 0), 1)

        else:
            print('no face detected!')


    cv.imshow('smile recognition', img)
    cv.waitKey(0)

cv.destroyAllWindows()

