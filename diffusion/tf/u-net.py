import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, \
    Conv2DTranspose


def encoder_block(inputs, n_filters, k_size, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, k_size, activation='relu', padding='same', kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, k_size, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection


def decoder_block(prev_layer_input, skip_layer_input, n_filters, k_size):
    up = Conv2DTranspose(n_filters, (k_size, k_size), strides=(2, 2), padding='same')(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)
    conv = Conv2D(n_filters, k_size, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters, k_size, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
    return conv


def u_net(input_shape=(256, 256, 3), nb_filters=32, k_size=3):
    inputs = keras.Input(shape=input_shape)

    encoder1, l1 = encoder_block(inputs, nb_filters, k_size, max_pooling=True)
    encoder2, l2 = encoder_block(encoder1, nb_filters * 2, k_size, max_pooling=True)
    encoder3, l3 = encoder_block(encoder2, nb_filters * 4, k_size, max_pooling=True)
    encoder4, l4 = encoder_block(encoder3, nb_filters * 8, k_size, max_pooling=False)

    bridge1 = Conv2D(nb_filters * 16, k_size, activation='relu', padding='same')(encoder4)
    bridge2 = Conv2D(nb_filters * 16, k_size, activation='relu', padding='same')(bridge1)

    decoder1 = decoder_block(bridge2, l4, nb_filters * 8, k_size)
    decoder2 = decoder_block(decoder1, l3, nb_filters * 4, k_size)
    decoder3 = decoder_block(decoder2, l2, nb_filters * 2, k_size)
    decoder4 = decoder_block(decoder3, l1, nb_filters, k_size)

    outputs = Conv2D(1, 1, activation='sigmoid')(decoder4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
