import keras
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, add, concatenate, merge
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


def basic_block(input_tensor):
    '''
    ----------
    basic block
    ----------
    '''
    # input tensor for a 3-channel 256x256 image
   
    y = Conv2D(12, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal')(input_tensor)
    y = Conv2D(12, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal')(y)
    y = Conv2D(12, (3, 3), padding='same')(y)
    # this returns x + y.
    z = add([y, input_tensor])
    z = BatchNormalization()(z)
    z = Activation('elu')(z)
    
    out = MaxPooling2D((2, 2))(z)
    return out


# ANN MODEL
x1 = Input((480, 480,1))
#x1 = MaxPooling2D((2, 2))(x1) # for the faster learning/prediction you can keep it
x1 = basic_block(x1)
x1 = MaxPooling2D((5, 5))(x1)
x2 = basic_block(x1)
x3 = Flatten()(x2)
x3 = Dropout(0.5)(x3)
x3 = Dense(10, activation='elu')(x3)
out = Dense(1, activation='sigmoid')(x3)


model = Model(inputs=x1, outputs=out)
print(model.summary())
