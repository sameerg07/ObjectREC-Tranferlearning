#!/usr/bin/env python3

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

def getModel(num_classes):
    # load vgg16 without dense layer and with theano dim ordering
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

    # number of classes in your dataset e.g. 20

    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    # create graph of your new model
    head_model = Model(input = base_model.input, output = predictions)

    # compile the model
    head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    head_model.summary()

    return head_model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def getDataGens(img_width, img_height):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,    
            rescale=1./255,
            fill_mode='nearest',
            horizontal_flip=True)

    val_datagen = ImageDataGenerator(
            rescale=1./255,
            fill_mode='constant',
            horizontal_flip=True)

    print("TEST DATA")
    train_generator = train_datagen.flow_from_directory(
            "./train",
            target_size=(img_width, img_height),
            batch_size=32,
            # color_mode="grayscale"
            )

    print("VALIDATION DATA")
    val_generator = val_datagen.flow_from_directory(
            "./val",
            target_size=(img_width, img_height),
            batch_size=32,
            # color_mode="grayscale"
            )

    return (train_generator, val_generator)

N = 224
spe=128
nb_epoch = 1000

# checkpointer = ModelCheckpoint(filepath="./tmp/tmp_model.h5", verbose=1, save_best_only=True)

model = getModel(2)
model.save("./models/init_model.hdf5")

tGen, vGen = getDataGens(N, N)

checkpoint = ModelCheckpoint(filepath="./models/checkpoint_model-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    tGen,
    samples_per_epoch=spe,
    nb_epoch=nb_epoch,
    validation_data=vGen,
    nb_val_samples=32,
    verbose=2,
    callbacks=callbacks_list)

model.save("./models/end_model.hdf5")