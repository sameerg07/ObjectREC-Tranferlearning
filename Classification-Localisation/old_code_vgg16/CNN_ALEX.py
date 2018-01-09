# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization

# #AlexNet with batch normalization in Keras 
# #input image is 224x224
# def CNN_ALEX(weights_path=None):
#     model_name = "ALEX"
#     model = Sequential()
#     model.add(Convolution2D(64,11, 11,3, border_mode='full'))
#     model.add(BatchNormalization((64,226,226)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(poolsize=(3, 3)))

#     model.add(Convolution2D(128,7, 7,64, border_mode='full'))
#     model.add(BatchNormalization((128,115,115)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(poolsize=(3, 3)))

#     model.add(Convolution2D(192,3, 3,128 border_mode='full'))
#     model.add(BatchNormalization((128,112,112)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(poolsize=(3, 3)))

#     model.add(Convolution2D(256,3, 3,192 border_mode='full'))
#     model.add(BatchNormalization((128,108,108)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(poolsize=(3, 3)))

#     model.add(Flatten())
#     model.add(Dense(12*12*256, 4096, init='normal'))
#     model.add(BatchNormalization(4096))
#     model.add(Activation('relu'))
#     model.add(Dense(4096, 4096, init='normal'))
#     model.add(BatchNormalization(4096))
#     model.add(Activation('relu'))
#     model.add(Dense(4096, 20, init='normal'))
#     model.add(BatchNormalization(20))
#     model.add(Activation('softmax'))
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy',optimizer=sgd,
#                   metrics=['accuracy']
#                  )
#     return model,model_name
# pass