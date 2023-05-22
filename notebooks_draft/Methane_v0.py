import os, folium, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import imageio
import geopandas as gpd
from IPython.display import Image, display
import tensorflow as tf
from tensorflow.keras import layers, callbacks, backend, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


labels = gpd.read_file(r'D:\New folder\labels.geojson')

labels=labels.iloc[:1000]

labels_unique = labels[(labels.Type == 'Negative') | (labels.Type == 'CAFOs') | (labels.Type == 'WWTreatment')
           | (labels.Type == 'Landfills') | (labels.Type == 'RefineriesAndTerminals')
           | (labels.Type == 'ProcessingPlants') | (labels.Type == 'Mines')]


X = []
for im_path in labels_unique.Image_Folder.values:
     X.append(imageio.imread(f'D:\\New folder\\{im_path}/naip.png',pilmode="RGB"))


# RGB Image of 3 samples
#fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12,5))
#axs[0].imshow(X[0][:,:,:3])
#axs[1].imshow(X[10][:,:,:3])
#axs[2].imshow(X[1000][:,:,:3])



#m = folium.Map([labels_unique.Latitude.min(), labels_unique.Longitude.min()], zoom_start=5, tiles='cartodbpositron')
#folium.GeoJson(labels_unique.geometry.iloc[0:200]).add_to(m)



mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels_unique.Type.values.reshape(labels_unique.Type.values.shape[0], 1))
X=np.array(X)
y=np.array(y)
np.save(r"D:\New folder\x_ndarray",X)
np.save(r"D:\New folder\y_ndarray",y)

#X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.7, stratify=y)
#X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=0.5, stratify=y_rest)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.7)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=0.5)


def invert_ecoding(encoded_labels, categories):
    categories = ['[unk]']+categories
    return np.take(categories, np.argwhere(encoded_labels == 1.0)[:,1])


y_tra_orig = np.argwhere(y_train == 1.0)[:,1]
y_res_orig = np.argwhere(y_rest == 1.0)[:,1]


def build_model():
    inputs = layers.Input(shape=(720, 720, 3))

    out = layers.Conv2D(16, (2, 2), activation='relu', padding='same')(inputs)
    out = layers.MaxPool2D()(out)
    out = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(out)
    out = layers.MaxPool2D()(out)
    out = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(out)
    out = layers.MaxPool2D()(out)
    out = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(out)
    out = layers.Flatten()(out)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(7, activation='sigmoid')(out)
    model = Model(inputs=inputs, outputs=out)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_model():
    i = layers.Input(shape=(720,720, 3), dtype = tf.float32)
    i1 = layers.Resizing(
        224,
        224,
        interpolation='bilinear',
        crop_to_aspect_ratio=False
    )(i)
    out = tf.keras.applications.DenseNet121(include_top=True,
                      weights='imagenet'
                      )(i1)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(7, activation='sigmoid')(out)
    model = Model(inputs=i, outputs=out)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



backend.clear_session()
model = build_model()
model.layers[1].trainable=False
model.summary()


cb = [callbacks.EarlyStopping(patience=5), callbacks.ReduceLROnPlateau(patience=3)]
epochs = 10
batch_size = 4
X_train=np.array(X_train)
X_val=np.array(X_val)
history = model.fit(X_train,
         y_train,
         epochs=epochs,
         batch_size=batch_size,
         callbacks=cb,
         validation_data=(X_val, y_val))

X_val=np.array(X_val)
len(X_train)
X_train[0].shape
len(y_train)

model.layers

