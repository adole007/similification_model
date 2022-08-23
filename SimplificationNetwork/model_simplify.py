from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, LeakyReLU
import tensorflow.keras.models as models
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta


def make_conv(i, f, k, s):
    o = Conv2D(f, k, strides=s, padding='same', kernel_initializer='he_normal')(i)
    o = BatchNormalization()(o)
    return LeakyReLU()(o)


def make_last_conv(i, f, k, s):
    return Conv2D(f, k, strides=s, padding='same', kernel_initializer='he_normal', activation='sigmoid')(i)


def make_deconv(i, f, k, s):
    o = Conv2DTranspose(f, k, strides=s, padding='same', kernel_initializer='he_normal')(i)
    o = BatchNormalization()(o)
    return LeakyReLU()(o)


def new_model(input_size=(512, 512, 1)):
    i = Input(input_size)

    d1 = make_conv(i, 48, (5, 5), (2, 2))
    f1 = make_conv(d1, 128, (3, 3), (1, 1))
    f2 = make_conv(f1, 128, (3, 3), (1, 1))

    d2 = make_conv(f2, 256, (3, 3), (2, 2))
    f3 = make_conv(d2, 256, (3, 3), (1, 1))
    f4 = make_conv(f3, 256, (3, 3), (1, 1))

    d3 = make_conv(f4, 256, (3, 3), (2, 2))
    f5 = make_conv(d3, 512, (3, 3), (1, 1))
    f6 = make_conv(f5, 1024, (3, 3), (1, 1))
    f7 = make_conv(f6, 1024, (3, 3), (1, 1))
    f8 = make_conv(f7, 1024, (3, 3), (1, 1))
    f9 = make_conv(f8, 1024, (3, 3), (1, 1))
    f10 = make_conv(f9, 512, (3, 3), (1, 1))
    f11 = make_conv(f10, 256, (3, 3), (1, 1))

    u1 = make_deconv(f11, 256, (4, 4), (2, 2))
    f12 = make_conv(u1, 256, (3, 3), (1, 1))
    f13 = make_conv(f12, 128, (3, 3), (1, 1))

    u2 = make_deconv(f13, 128, (4, 4), (2, 2))
    f14 = make_conv(u2, 128, (3, 3), (1, 1))
    f15 = make_conv(f14, 48, (3, 3), (1, 1))

    u3 = make_deconv(f15, 48, (4, 4), (2, 2))
    f16 = make_conv(u3, 24, (3, 3), (1, 1))
    o = make_last_conv(f16, 1, (3, 3), (1, 1))

    model = Model(inputs=i, outputs=o)

    return model


def simplify(pretrained_weights=None, input_size=(512, 512, 1)):
    if pretrained_weights:
        model = models.load_model(pretrained_weights)
    else:
        model = new_model(input_size)

    loss_function = mean_squared_error
    metrics = ['mean_squared_error']
    model.compile(optimizer=Adadelta(), loss=loss_function, metrics=metrics)
    model.summary()

    return model



