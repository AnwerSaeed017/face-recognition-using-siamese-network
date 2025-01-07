from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization, Dropout
from keras.layers import Lambda, Subtract
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, zeros
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K

# Initialize weights as in paper
kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)

# Initialize bias as in paper
bias_initializer = tf.keras.initializers.Constant(value=0.5)

def W_init(shape, dtype=None, name=None):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0, scale=1e-2, size=shape)
    return tf.Variable(values, dtype=dtype, name=name)


def b_init(shape, dtype=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return tf.Variable(values, dtype=dtype)

input_shape = (100, 100, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
negative = Input(input_shape)

# Define initializers
kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
bias_initializer = RandomNormal(mean=0.5, stddev=0.01)

convnet = tf.keras.Sequential()
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                   kernel_initializer=kernel_initializer, kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())


convnet.add(Conv2D(128, (7, 7), activation='relu',
                   kernel_regularizer=l2(2e-4), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
convnet.add(MaxPooling2D())


convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(2e-4), bias_initializer=bias_initializer))
convnet.add(MaxPooling2D())


convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(2e-4), bias_initializer= bias_initializer))
convnet.add(Flatten())


convnet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3),
                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss
# Encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
encoded_n = convnet(negative)
prediction = Lambda(triplet_loss)([encoded_l,encoded_r,encoded_n])
siamese_net = Model(inputs=[left_input, right_input,negative], outputs=prediction)



def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

optimizer = Adam(0.00006)
siamese_net.compile(loss=identity_loss, optimizer=optimizer)

siamese_net.summary()