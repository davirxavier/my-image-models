from enum import Enum
from typing import Literal, Tuple

import tensorflow.nn
from keras import Model
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, ReLU, DepthwiseConv2D, \
    Activation

DOWNSAMPLE_CONV = "conv"
DOWNSAMPLE_MAX_POOL = "max_pool"
EXTRA_DEPTHWISE = "depthwise"
EXTRA_FULL_CONV = "extra_full"
EXTRA_CONV_1x1 = "conv_1x1"

def build_model(input_shape, alpha, num_classes, weights, conv_blocks=((16, DOWNSAMPLE_MAX_POOL), (32, DOWNSAMPLE_MAX_POOL), (64, DOWNSAMPLE_MAX_POOL), (128, DOWNSAMPLE_MAX_POOL)), dropout_rate=0.2, include_head=True):
    def gw(w):
        return int(w * alpha)

    bc = 0
    def conv_block(x, w, downsample, extra):
        nonlocal bc
        if downsample not in {DOWNSAMPLE_CONV, DOWNSAMPLE_MAX_POOL, None}:
            raise ValueError(f"Invalid downsample mode: {downsample}")

        conv_strides = (2, 2) if downsample == DOWNSAMPLE_CONV else (1, 1)
        aw = gw(w)
        is_depthwise = EXTRA_DEPTHWISE in extra

        if is_depthwise:
            x = DepthwiseConv2D((3, 3), strides=conv_strides, name=f"block{bc}_depth_conv_{aw}", padding="same")(x)
            x = BatchNormalization(name=f"block_{bc}_depth_bn")(x)
            x = Activation(tensorflow.nn.relu6, name=f"block_{bc}_depth_relu6")(x)

        kernel = 1 if is_depthwise else 3
        if EXTRA_CONV_1x1 in extra:
            kernel = 1

        x = Conv2D(aw, kernel,
                   name=f"block_{bc}_conv_{aw}_kn_{kernel}",
                   padding="same",
                   strides=1 if is_depthwise else conv_strides)(x)

        x = BatchNormalization(name=f"block_{bc}_bn")(x)
        x = Activation(tensorflow.nn.relu6, name=f"block_{bc}_relu6")(x)

        if downsample == DOWNSAMPLE_MAX_POOL:
            x = MaxPooling2D((2, 2), name=f"block_{bc}_pool")(x)

        if EXTRA_FULL_CONV in extra:
            x = Conv2D(aw, (3, 3),
                       name=f"block_{bc}_conv_{aw}_kn_3_2",
                       padding="same",
                       strides=1)(x)
            x = BatchNormalization(name=f"block_{bc}_2_bn")(x)
            x = Activation(tensorflow.nn.relu6, name=f"block_{bc}_2_relu6")(x)

        bc += 1
        return x

    input = InputLayer(input_shape, name="input").input
    x = input

    for block_conf in conv_blocks:
        if isinstance(block_conf, tuple):
            bw = block_conf[0]
            ds = block_conf[1]

            if len(block_conf) > 2:
                ex = block_conf[2]
            else:
                ex = []
        else:
            bw = block_conf
            ds = None
            ex = []

        x = conv_block(x, bw, ds, ex)

    if include_head:
        x = Flatten(name="output_flatten")(x)
        x = Dropout(dropout_rate, name="output_dropout")(x)
        x = Dense(gw(32), activation="relu", name="output_dense")(x)
        output = Dense(num_classes, activation="softmax", name="output_classes")(x)
    else:
        output = x

    model = Model(inputs=input, outputs=output)

    if weights is not None:
        model.load_weights(weights)

    return model