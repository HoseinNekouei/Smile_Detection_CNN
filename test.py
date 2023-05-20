import numpy as np
import cv2 as cv
from mtcnn import MTCNN
from keras.models import load_model

detector = MTCNN()
REDCOLOR = (0, 0, 255)
GREENCOLOR = (0, 255, 0)
smile_net = load_model(r'deep_learning\Week6\smile_detector\smile_model.h5')

img = cv.imread(r'deep_learning\Week6\smile_detector\test\02.jpg')
frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
out = detector.detect_faces(frame_rgb)

x, y, w, h = out[0]['box']
face = img[y:y+h, x:x+w]

face = cv.resize(face, (32, 32))
face = face / 255

face = np.array([face])

predict = smile_net.predict(face)[0][0]
predict = np.round(predict)

print(predict)
if predict == 1.0:
    COLOR = GREENCOLOR
    label = 'smile'
else:
    COLOR = REDCOLOR
    label = 'no_smile'

cv.rectangle(img, (x, y), (x+w, y+h), COLOR, 2)

font = cv.QT_FONT_NORMAL
# get text size
(text_width, text_heigh), baseline = cv.getTextSize(label, font, 1, 1)
# draw a rectangle above object rectangle
cv.rectangle(img, (x, y), (x-5+text_width, y -
                text_heigh-baseline), COLOR, thickness=-1)

cv.putText(img,label, (x, y-6),
            font, 1, (0, 0, 0), 1)

cv.imshow('Smile Detection', img)
cv.waitKey(0)

cv.destroyAllWindows()