    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    # Only because git push is not working

    def conv_block(x, filters, dropout_rate=0.0):
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        return x

    def upsample_concat(x_deeper, x_skip, filters):
        x = UpSampling2D()(x_deeper)
        x = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal')(x)
        return concatenate([x, x_skip])


    def UNetPP(input_shape=(512,512, 1), last_activation='sigmoid'):
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
        x50 = MaxPooling2D()(x40)
        x50 = conv_block(x50, 1024, 0.5)


        # Decoder (Nested Connections)
        x01 = conv_block(upsample_concat(x10, x00, 32), 32, 0.1)
        x11 = conv_block(upsample_concat(x20, x10, 64), 64, 0.2)
        x02 = conv_block(concatenate([x00, x01, upsample_concat(x11, x00, 32)]), 32, 0.1)

        x21 = conv_block(upsample_concat(x30, x20, 128), 128, 0.3)
        x12 = conv_block(concatenate([x10, x11, upsample_concat(x21, x10, 64)]), 64, 0.2)
        x03 = conv_block(concatenate([x00, x01, x02, upsample_concat(x12, x00, 32)]), 32, 0.1)

        x31 = conv_block(upsample_concat(x40, x30, 256), 256, 0.4)
        x22 = conv_block(concatenate([x20, x21, upsample_concat(x31, x20, 128)]), 128, 0.3)
        x13 = conv_block(concatenate([x10, x11, x12, upsample_concat(x22, x10, 64)]), 64, 0.2)
        x04 = conv_block(concatenate([x00, x01, x02, x03, upsample_concat(x13, x00, 32)]), 32, 0.1)

        x41 = conv_block(upsample_concat(x50, x40, 512), 512, 0.5)
        x32 = conv_block(concatenate([x30, x31, upsample_concat(x41, x30, 256)]), 256, 0.4)
        x23 = conv_block(concatenate([x20, x21, x22, upsample_concat(x32, x20, 128)]), 128, 0.3)
        x14 = conv_block(concatenate([x10, x11, x12, x13, upsample_concat(x23, x10, 64)]), 64, 0.2)
        x05 = conv_block(concatenate([x00, x01, x02, x03, x04, upsample_concat(x14, x00, 32)]), 32, 0.1)

        # Output
        output = Conv2D(1, 1, activation=last_activation, dtype = 'float32')(x05)

        model = Model(inputs, output)
        return model