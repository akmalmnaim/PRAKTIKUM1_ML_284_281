# Pneumonia X-ray Classification using CNN

| Anggota             | NIM             |
| ------------------- | --------------- |
| Akmal Muhammad Naim | 201810370311284 |
| Qhistina Dyah K     | 201810370311281 |

- Team-Details : [Team-details.xlsx](https://docs.google.com/spreadsheets/d/1MFoxcR4i5nY0az9baZ_RGuibZcmJ-pqobErTnSoZ8BI/edit#gid=0)
- Sprint-Details : [Sprint-details.xlsx](https://docs.google.com/spreadsheets/d/1YglOE6kn-aNgLgnFz80dmph8DgTlm3rExu2rakgyuZQ/edit?usp=sharing)

Praktikum Machine Learning 201810370311284 & 201810370311281

Link Dataset : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Konten

- [Informasi](#general-information)
- [Library](#technologies-used)
- [Features](#features)
- [Screenshots](#screenshots)
- [Setup](#setup)
- [Model](#model)
- [Project Status](#project-status)
- [Room for Improvement](#room-for-improvement)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
<!-- * [License](#license) -->

## General Information

- Dataset yang digunakan berjudul "Lung and Colon Cancer Histopathological Images" yang merupakan dataset citra.
- Jumlah dataset 5856 files.
- Terbagi dalam 2 class

## Library

- Tensorflow 2.0
- Keras
- Matplotlib
- Numpy

## Features

#Kelas datasetnya meliputi :
-Normal
Pneumonia

## Screenshots

![Example screenshot](https://i.imgur.com/jZqpV51.png)

## Setup

### prepocessing step :

- Splitting data 80:19:1 (80 train, 19 test, 1 validation)

### Augmentation :

- ImageDataGenerator :
  ** rescale : 1/255
  ** rotation range : 40
  ** width and height shift range : 0.2
  ** zoom range : 0.2
  ** horizontal flip
  ** fill mode : nearest
  \*\* color mode : RGB
- hyper parameter : Optimizer(adam), learningrate(0,000001), loss(binary_crossentrophy)


## Model :

- Model 1 : Layer, Dense, Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Dropout, Flatten
- Activation : relu (Without Augmentation)

```
# Feature Extraction Layer
model = Sequential()

model.add(InputLayer(input_shape=[150,150,3]))
model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

```

- model sequential 2 : Layer, Dense, Conv2D, AveragePool2D, Flatten, BatchNormalization, Dropout, AveragePool2D
- Activation : relu (With Augmentation)

```
model2 = Sequential()

model2.add(InputLayer(input_shape=[250,250,3]))
model2.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='elu'))
model2.add(BatchNormalization())
model2.add(AveragePool2D(pool_size=2, padding='same'))
model2.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='elu'))
model2.add(BatchNormalization())
model2.add(AveragePool2D(pool_size=2, padding='same'))
model2.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='elu'))
model2.add(BatchNormalization())
model2.add(AveragePool2D(pool_size=2, padding='same'))
model2.add(Dropout(0.25))
model2.add(Flatten())

model2.add(Dense(128, activation='elu'))
model2.add(Dropout(0.5))
model2.add(Dense(5, activation='softmax'))

```

- model Hyperparameter Tuning 2 : Layer, Dense, Conv2D, AveragePool2D, Flatten, BatchNormalization, Dropout, AveragePool2D
- Activation : relu (With Augmentation)

hyperparameter tuning

```
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.4, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'RMSProp']))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu', 'sigmoid']))
```

```
model = tf.keras.models.Sequential([

                                      tf.keras.layers.Conv2D(64, (5, 5), activation=hparams[HP_ACTIVATION],
                                      input_shape=(128,128,3)),
                                      tf.keras.layers.BatchNormalization(),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),

                                      tf.keras.layers.Conv2D(64, (5, 5), activation=hparams[HP_ACTIVATION]),
                                      tf.keras.layers.BatchNormalization(),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),

                                      tf.keras.layers.Conv2D(32, (5, 5), activation=hparams[HP_ACTIVATION]),
                                      tf.keras.layers.BatchNormalization(),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                                      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
                                      tf.keras.layers.Flatten(),


                                      tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=hparams[HP_ACTIVATION]),
                                      tf.keras.layers.Dense(64, activation=hparams[HP_ACTIVATION]),
                                      tf.keras.layers.Dense(32, activation=hparams[HP_ACTIVATION]),
                                      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
                                      tf.keras.layers.Dense(2, activation=tf.nn.softmax),
                                      ])


```

- model Sequential 3 : Layer, Dense, Conv2D, AveragePool2D, Flatten, BatchNormalization, Dropout, AveragePool2D
- Activation : relu (With Augmentation)

```
model = Sequential()

model.add(InputLayer(input_shape=[150,150,3]))
model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())


```

- model Pretrained VGG19 4 : Layer, Dense, Conv2D, AveragePool2D, Flatten, BatchNormalization, Dropout, AveragePool2D
- Activation : relu (With Augmentation)

```
baseModel = VGG16(include_top=False, input_tensor=Input(shape=(150, 150, 3)), weights= 'imagenet')
```

Fully Connected Layer

```
class FCHeadNet:
  def build(baseModel, classes, D):
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(D, activation='relu')(headModel)
    headModel = Dropout(0.7)(headModel)
    headModel = Dense(D, activation='relu')(headModel)
    headModel = Dropout(0.7)(headModel)
    headModel = Dense(classes, activation='sigmoid')(headModel)
    return headModel


```
