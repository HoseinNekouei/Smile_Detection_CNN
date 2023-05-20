import numpy as np
import glob as gb
import cv2 as cv
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import models, layers
from keras.optimizers import SGD


detector = MTCNN()
BATCHSIZE = 32
EPOCHS = 30


def data_load_preprocessing():
    datalist = []
    labels = []

    for index, item in enumerate(gb.glob(r'deep_learning\Week6\smile_detector\dataset\*\*')):
        img = cv.cvtColor(cv.imread(item), cv.COLOR_BGR2RGB)
        out = detector.detect_faces(img)

        if len(out) > 0:
            x, y, w, h = out[0]['box']
            # cofidence = out[0]['cofidence']
            face_ROI = img[y:y+h, x:x+w]
            face = cv.resize(face_ROI, (32, 32))

            face = face / 255.0
            datalist.append(face)

            label = item.split('\\')[-2]
            labels.append(label)
        else:
            continue

    datalist = np.array(datalist)

    x_train, x_test, y_train, y_test = train_test_split(
        datalist, labels, test_size=0.2, random_state=42)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    return x_train, x_test, y_train, y_test


def neural_network():
    opt = SGD(learning_rate=0.01, decay=0.00025)

    net = models.Sequential([layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), input_dim=3072),
                             layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                             layers.BatchNormalization(),
                             layers.MaxPool2D(),

                             layers.Conv2D(64, (3, 3), activation='relu'),
                             layers.Conv2D(64, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.MaxPool2D(),

                             layers.Flatten(),

                             layers.Dense(32, activation='relu'),
                             layers.Dropout(0.25),
                             layers.BatchNormalization(),

                             layers.Dense(1, 'sigmoid'),
                             ])
    print(net.summary())

    net.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    H = net.fit(x_train, y_train, batch_size=BATCHSIZE,
                epochs=EPOCHS, validation_data=(x_test, y_test))

    loss, accuracy = net.evaluate(x_test, y_test)
    print(f'Test loss: {loss :.2f}, Test accuracy: {accuracy :.2f}')

    net.save(r'deep_learning\Week6\smile_detector\smile_model.h5')

    return H


def show_result():
    plt.style.use('ggplot')

    plt.plot(np.arange(EPOCHS), model.history['loss'], label='Train loss')
    plt.plot(np.arange(EPOCHS),
             model.history['accuracy'], label='Train accuracy')
    plt.plot(np.arange(EPOCHS), model.history['val_loss'], label='Test loss')
    plt.plot(np.arange(EPOCHS),
             model.history['val_accuracy'], label='Test accuracy')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.title('smile dataset Training')

    plt.show()


x_train, x_test, y_train, y_test = data_load_preprocessing()
model = neural_network()
show_result()
