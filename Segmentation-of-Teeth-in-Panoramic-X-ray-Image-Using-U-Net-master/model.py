from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def conv_block(x, filters, dropout_rate=0.0):
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x

def UNetPP(input_shape=(128, 128, 1), last_activation='sigmoid'):
    inputs = Input(input_shape)

    # Encoder
    x00 = conv_block(inputs, 32, 0.1)
    x10 = MaxPooling2D()(x00)
    x10 = conv_block(x10, 64, 0.2)
    x20 = MaxPooling2D()(x10)
    x20 = conv_block(x20, 128, 0.3)
    x30 = MaxPooling2D()(x20)
    x30 = conv_block(x30, 256, 0.4)
    x40 = MaxPooling2D()(x30)
    x40 = conv_block(x40, 512, 0.5)

    # Decoder (Nested Connections)
    x01 = conv_block(concatenate([x00, UpSampling2D()(x10)]), 32, 0.1)
    x11 = conv_block(concatenate([x10, UpSampling2D()(x20)]), 64, 0.2)
    x02 = conv_block(concatenate([x00, x01, UpSampling2D()(x11)]), 32, 0.1)

    x21 = conv_block(concatenate([x20, UpSampling2D()(x30)]), 128, 0.3)
    x12 = conv_block(concatenate([x10, x11, UpSampling2D()(x21)]), 64, 0.2)
    x03 = conv_block(concatenate([x00, x01, x02, UpSampling2D()(x12)]), 32, 0.1)

    x31 = conv_block(concatenate([x30, UpSampling2D()(x40)]), 256, 0.4)
    x22 = conv_block(concatenate([x20, x21, UpSampling2D()(x31)]), 128, 0.3)
    x13 = conv_block(concatenate([x10, x11, x12, UpSampling2D()(x22)]), 64, 0.2)
    x04 = conv_block(concatenate([x00, x01, x02, x03, UpSampling2D()(x13)]), 32, 0.1)

    # Output
    output = Conv2D(1, 1, activation=last_activation)(x04)

    model = Model(inputs, output)
    return model