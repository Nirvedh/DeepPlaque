from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

from losses import (
    binary_crossentropy,
    dice_loss,
    bce_dice_loss,
    dice_coef,
    weighted_bce_dice_loss,
    box_dice_loss
)


def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu'):
    skip = []
    for i in range(n_block):
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    dil_f =[1,2,3,4,5,6] #this is for dilated unet
    #dil_f =[1,1,1] #this for regular unet remeber to change depth=3 and n_blocks=4 when using this
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=dil_f[i])(x)
            x = BatchNormalization()(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu'):
    for i in reversed(range(n_block)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip[i], x])
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)

    return x


def get_dilated_unet(
        input_shape=(1920, 1280, 3),
        mode='cascade',
        filters=44,
        n_block=3,
        lr=0.0001,
        loss=binary_crossentropy,#bce_dice_loss,
        n_class=1
):
    inputs = Input(input_shape)
    
    enc, skip = encoder(inputs, filters, n_block)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model
