import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import class_weight

import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# from keras.applications import VGG16
# Load the VGG model
vgg_conv = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


train_images = np.load("processed_data/train_total.npy")
train_labels = np.load("processed_data/train_total_label.npy")



class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)
train_labels = keras.utils.to_categorical(train_labels, 58)

# separate into train and valid sets
train_images = train_images[0:499962]
train_labels = train_labels[0:499962]

valid_images = train_images[-166653:0]
valid_labels = train_labels[-166653:0]



#data augmentation

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# fit the data augmentation
train_datagen.fit(train_images[0:100000])
print("Part 1 loaded")
train_datagen.fit(train_images[100000:200000])
print("Part 2 loaded")
train_datagen.fit(train_images[200000:300000])
print("Part 3 loaded")
train_datagen.fit(train_images[300000:400000])
print("Part 4 loaded")
train_datagen.fit(train_images[400000:])
print("All parts loaded")

validation_datagen.fit(valid_images[0:50000])
validation_datagen.fit(valid_images[50000:100000])
validation_datagen.fit(valid_images[100000:])

# Takes data & label arrays, generates batches of augmented data.

train_generator = train_datagen.flow(train_images,train_labels, batch_size=100)
validation_generator = validation_datagen.flow(valid_images,valid_labels, batch_size=100)

'''train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=train_batchsize,
        class_mode='categorical')'''



# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


NAME = "VGG-ADAM5-Generator2"

'''class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_labels),
                                                 train_labels)'''

tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))

# add early-stopping: stop if there is no improvement on val_loss more than 10 epochs
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')

# add reducelronplateau: halve the learning_rate if there is no improvement on val_loss more than 5 epochs
reducelronplateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0,
                                                      mode='auto')
checkpointfunc = keras.callbacks.ModelCheckpoint('trained_models/{}.model'.format(NAME), monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 save_weights_only=False, mode='min', period=1)
val_acc = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: f(epoch, logs['val_acc']))
val_loss = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: f(epoch, logs['val_loss']))


model = keras.Sequential()
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, size=(48, 48))))
# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(200, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(keras.layers.Conv2D(180, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Conv2D(180, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(keras.layers.Conv2D(140, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(180, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# from fine_tune
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(58, activation='softmax'))

# model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])


# load previous trained model for lateron initial_epoch
'''if os.path.exists('trained_models/{}.model'.format(NAME)):
    model = tf.keras.models.load_model('trained_models/{}.model'.format(NAME))
    print("checkpoint_loaded")'''


# Train the model
model.fit_generator(
      train_generator,
      steps_per_epoch=len(train_images)//train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=len(valid_images)//validation_generator.batch_size, class_weight=class_weights,
      verbose=1, callbacks=[tensorboard, checkpointfunc, earlystopping, reducelronplateau, val_acc, val_loss])



'''model.fit(train_images, train_labels, batch_size=100, epochs=35, initial_epoch=0, validation_split=0.20,
          shuffle=True, class_weight=class_weights,
          callbacks=[tensorboard, checkpointfunc, earlystopping, reducelronplateau])'''

model.save('trained_models/{}.model'.format(NAME))