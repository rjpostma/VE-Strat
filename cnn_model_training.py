# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import layers
from keras.models import load_model
import keras.utils
from keras.models import Model
import tensorflow
from keras import backend as K
from tensorflow import data as tf_data


#%%
print(tensorflow.test.is_built_with_gpu_support())
print(tensorflow.config.list_physical_devices('GPU'))


#%%
image_size = (64, 64)
batch_size = 64

train_ds = keras.utils.image_dataset_from_directory(
    "TRAINING/ROIs_TRAINING/",
    label_mode="categorical",
    color_mode='grayscale',
    validation_split=0.1,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    )

val_ds = keras.utils.image_dataset_from_directory(
    "TRAINING/ROIs_TRAINING/",
    label_mode="categorical",
    color_mode='grayscale',
    validation_split=0.1,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    )


data_augmentation_layers = [
    layers.RandomFlip("horizontal_and_vertical"),
    tensorflow.keras.layers.RandomContrast(factor=0.2),
    tensorflow.keras.layers.RandomBrightness(factor=0.1,value_range=[0.0, 1.0])
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


#%%

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)




#%%
inputs = layers.Input(shape=(64, 64, 1))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(128, 5, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

previous_block_activation = x

for size in [128, 256, 512]:
     x = layers.Activation("relu")(x)
     x = layers.SeparableConv2D(size, 3, padding="same")(x)
     x = layers.BatchNormalization()(x)

     x = layers.Activation("relu")(x)
     x = layers.SeparableConv2D(size, 3, padding="same")(x)
     x = layers.BatchNormalization()(x)

     x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
     
     residual = layers.Conv2D(size, 1, strides=2, padding="same")(
         previous_block_activation
     )
     x = layers.add([x, residual])
     previous_block_activation = x

x = layers.SeparableConv2D(32, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

flattened  = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(flattened)
outputs = layers.Dense(5, activation='softmax')(x)

classifier = Model(inputs, outputs)
keras.utils.plot_model(classifier, show_shapes=True)
classifier.summary()

keras.utils.plot_model(classifier, show_shapes=True)



#%%
checkpoint_filepath = 'checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

classifier.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(name='acc')],
)

history = classifier.fit(
    train_ds,
    epochs=40,
    callbacks=[model_checkpoint_callback],
    validation_data=val_ds,
)


#%%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_model.pdf')



#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_model.pdf')



#%%
encoder = Model(inputs, flattened)
encoder.save("model-encoder.keras")


