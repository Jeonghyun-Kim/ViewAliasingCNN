from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Input, concatenate, add, MaxPooling2D, UpSampling2D, Dense, Flatten
import h5py
import numpy as np

def CNNClassifier(k):
    model = Sequential()
    model.add(Conv2D(k, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(k, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(k, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(k, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(k, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(k, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

def DenseResNet14(k):
    # Create Model
    # k = coefficient of depth of feature maps

    x_in = Input(shape=(None,None,1), name='input')

    # Feature Extraction
    skip_x1 = Conv2D(2*k, (5,5), padding='same', name='conv0')(x_in)

    # 1st conv block
    x = BatchNormalization()(skip_x1)
    x = Activation('relu')(x)
    x = Conv2D(k, (1,1), padding='same', name = 'conv1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_2')(x)

    # 2nd conv block
    skip_x2 = concatenate([x,skip_x1], axis = -1) # channel_num = 2k+k = 3k
    x = BatchNormalization()(skip_x2)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (1,1), padding='same', name = 'conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2_2')(x)

    # 3rd conv block
    skip_x3 = concatenate([x,skip_x2], axis = -1) # channel_num = 4k
    x = BatchNormalization()(skip_x3)
    x = Activation('relu')(x)
    x = Conv2D(3*k, (1,1), padding='same', name = 'conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3_2')(x)

    # 4th conv block
    skip_x4 = concatenate([x,skip_x3], axis = -1) # channel_num = 5k
    x = BatchNormalization()(skip_x4)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1,1), padding='same', name = 'conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4_2')(x)

    # 5th conv block
    skip_x5 = concatenate([x,skip_x4], axis = -1) # channel_num = 6k
    x = BatchNormalization()(skip_x5)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1,1), padding='same', name = 'conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5_2')(x)

    # 6th conv block
    skip_x6 = concatenate([x,skip_x5], axis = -1) # channel_num = 7k
    x = BatchNormalization()(skip_x6)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (1,1), padding='same', name = 'conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6_2')(x)

    # Aggregation
    x = concatenate([x,skip_x6], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv7')(x)

    # Residual(i.e. Noise) Learning
    x = add([x,x_in], name = 'output')

    model = Model(inputs=x_in, outputs=x)

    return model

def UNet24(k):

    x_in = Input(shape=(None,None,1), name='input')

#     x = BatchNormalization()(x_in)
#     x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_1')(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x1 = Conv2D(k, (3,3), padding='same', name = 'conv1_3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x2 = Conv2D(2*k, (3,3), padding='same', name = 'conv2_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x3 = Conv2D(4*k, (3,3), padding='same', name = 'conv3_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x4 = Conv2D(8*k, (3,3), padding='same', name = 'conv4_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16*k, (3,3), padding='same', name = 'conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(16*k, (3,3), padding='same', name = 'conv5_2')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv5_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2DTranspose(8*k, (3,3), strides=(2, 2), padding='same', name = 'upconv6_1')(x)
    # x = Conv2DTranspose(8*k, (2,2), strides=(2, 2), padding='valid', name = 'upconv6_1')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = concatenate([x,skip_x4], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(8*k, (3,3), padding='same', name = 'conv6_2')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv6_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2DTranspose(4*k, (3,3), strides=(2, 2), padding='same', name = 'upconv7_1')(x)
    # x = Conv2DTranspose(4*k, (2,2), strides=(2, 2), padding='valid', name = 'upconv7_1')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = concatenate([x,skip_x3], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv7_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(4*k, (3,3), padding='same', name = 'conv7_2')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv7_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2DTranspose(2*k, (3,3), strides=(2, 2), padding='same', name = 'upconv8_1')(x)
    # x = Conv2DTranspose(2*k, (2,2), strides=(2, 2), padding='valid', name = 'upconv8_1')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = concatenate([x,skip_x2], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv8_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(2*k, (3,3), padding='same', name = 'conv8_2')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv8_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2DTranspose(k, (3,3), strides=(2, 2), padding='same', name = 'upconv9_1')(x)
    # x = Conv2DTranspose(k, (2,2), strides=(2, 2), padding='valid', name = 'upconv9_1')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = concatenate([x,skip_x1], axis = -1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv9_3')(x)

    x = add([x,x_in], name = 'output') # Residual Learning
    model = Model(inputs=x_in, outputs=x)

    return model

def Plain14(k):
    x_in = Input(shape=(None,None,1), name='input')

    x = Conv2D(k, (3,3), padding='same', name = 'conv1')(x_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv7')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv8')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv10')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv11')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv12')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv13')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv14')(x)

    model = Model(inputs=x_in, outputs=x)

    return model

def ResNet14(k):
    x_in = Input(shape=(None,None,1), name='input')

    x_0 = Conv2D(k, (3,3), padding='same', name = 'conv0')(x_in)

    # residual block 1
    x = BatchNormalization()(x_0)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_2')(x)
    x_1 = add([x,x_0]) # Skip connection


    # residual block 2
    x = BatchNormalization()(x_1)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2_2')(x)
    x_2 = add([x,x_1]) # Skip connection


    # residual block 3
    x = BatchNormalization()(x_2)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3_2')(x)
    x_3 = add([x,x_2]) # Skip connection

    # residual block 4
    x = BatchNormalization()(x_3)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4_2')(x)
    x_4 = add([x,x_3]) # Skip connection

    # residual block 5
    x = BatchNormalization()(x_4)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5_2')(x)
    x_5 = add([x,x_4]) # Skip connection

    # residual block 6
    x = BatchNormalization()(x_5)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6_2')(x)
    x_6 = add([x,x_5]) # Skip connection

    # Aggregation
    x = BatchNormalization()(x_6)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv7')(x)

    # Residual(i.e. Noise) Learning
    x = add([x,x_in], name = 'output')

    model = Model(inputs=x_in, outputs=x)

    return model
