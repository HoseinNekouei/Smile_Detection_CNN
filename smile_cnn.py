import numpy as np
import glob as gb
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


EPOCHS = 30
BATCHSIZE = 32
haar_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


def data_load_preprocessing():
    datalist = []
    labels = []

    for index, item in enumerate(gb.glob(r'dataset\*\*')):
        try:
            img = cv.imread(item)

            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Applying the face detection method on the grayscale image
            x, y, w, h = haar_cascade.detectMultiScale(gray_img, 1.1, 9)[0][0:]

            face_ROI = img[y:y+h, x:x+w]

            face = cv.resize(face_ROI, (64, 64))

            face = face / 255.0
            datalist.append(face)

            label = item.split('\\')[-2]
            labels.append(label)

        except:
            pass

        if index % 100 == 0:
            print(f'[info] {index}/3516 images processed!')

    datalist = np.array(datalist)

    # print(len(datalist))
    x_train, x_test, y_train, y_test = train_test_split(
        datalist, labels, test_size=0.2, random_state=42)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    return x_train, x_test, y_train, y_test


def neural_network():
    # define image generator (data augmentation)
    aug = ImageDataGenerator(rotation_range=20,
                             fill_mode='reflect')
    
    opt = SGD(learning_rate=0.01, decay=0.00025)

    net = models.Sequential([layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3), input_dim=12288),
                             layers.BatchNormalization(),
                             layers.Dropout(0.25),
                             layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                             layers.BatchNormalization(),
                             layers.MaxPool2D(),
                             layers.Dropout(0.25),

                             layers.Conv2D(64, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.Dropout(0.25),
                             layers.Conv2D(64, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.MaxPool2D(),
                             layers.Dropout(0.25),

                             layers.Conv2D(128, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.Dropout(0.25),
                             layers.Conv2D(128, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.Dropout(0.25),
                             layers.Conv2D(128, (3, 3), activation='relu'),
                             layers.BatchNormalization(),
                             layers.MaxPool2D(),
                             layers.Dropout(0.25),

                             layers.Flatten(),

                             layers.Dense(1024, activation='relu'),
                             layers.BatchNormalization(),
                             layers.Dropout(0.50),

                             layers.Dense(1, 'sigmoid'),
                             ])
    print(net.summary())

    net.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    H = net.fit(aug.flow(x_train, y_train, batch_size=BATCHSIZE),
                steps_per_epoch=len(x_train)//BATCHSIZE,
                epochs=EPOCHS, validation_data=(x_test, y_test))

    loss, accuracy = net.evaluate(x_test, y_test)
    print(f'Test loss: {loss :.2f}, Test accuracy: {accuracy :.2f}')

    net.save(r'smile_model.h5')

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
