from keras.models import Sequential
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
import json
import h5py
import os
import _pickle as cPickle
# import cPickle
import sys
import shutil
from keras import backend as K
K.set_image_dim_ordering('tf')

# import of custom functions
from vgg_train import VGG_16 as VGG_16_mod
from CNN_12 import New_model as CNN_12_mod
from CNN_12_DO import New_model as CNN_12_DO_mod

# folder = './work'
# train = folder + '/g_train'val = folder + '/g_val'

nb_epoch = 10
img_height, img_width = 224,224

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

train_generator = train_datagen.flow_from_directory(
        "./train",
        target_size=(img_width, img_height),
        batch_size=32 
        )

val_generator = val_datagen.flow_from_directory(
        "./val",
        target_size=(img_width, img_height),
        batch_size=32 
        )

print(train_generator.num_classes)

# exit()

# Specify the model used here.
model, model_name = VGG_16_mod()

# saves the temp model output in the model_dir - here the metric is least validation loss 
model_dir = './' + model_name + '/'
try:
    os.mkdir(model_dir)
except:
    pass
temp_path = model_dir + 'final_temp_model.h5'

print("XXXXXXXXXXXXXXXXXXXXXXXXx")

print(model.summary())

#update the model if validation loss reduces
checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)

print("XXXXXXXXXXXXXXXXXXXXXXXXx")

# this actually fits the model
output = model.fit_generator(
                                train_generator,
                                samples_per_epoch=32,
                                nb_epoch=nb_epoch,
                                validation_data=val_generator,
                                nb_val_samples=32,
                                callbacks=[checkpointer],
                                verbose=1)
print("XXXXXXXXXXXXXXXXXXXXXXXXx")

# Saves the final model weights at the end of the training
model.save(model_dir + 'final_final_model.h5')

print("XXXXXXXXXXXXXXXXXXXXXXXXx")
hist = output.history
params = output.params


model_json = model.to_json()
with open(model_dir + "final_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("final_weights_file.h5")
print("Saved model to disk") 
# save parameters and histor of the model
# History saves the Training accuracy and loss also Validation loss and accuracy
# Parameters saves the parmeter used and their values such as metrics saved during the training, the number of epochs etc.
#with open(model_dir + 'history.json', 'wb') as h:
 #   json.dump(hist, h)
#with open(model_dir + 'params.json', 'wb') as p:
#    json.dump(params, p)

# this saves the model architecture and the model weights.
# can be loaded later and used for prediction.    
# serialize model to JSON
   


