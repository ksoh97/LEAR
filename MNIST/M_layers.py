import tensorflow as tf
import tensorflow.keras as keras
l = tf.keras.layers
W, H = 28, 28
c = 10

def input_layer(input_shape=(W, H, 1), name=None):
    return l.Input(shape=input_shape)

def c1(input_shape=(14,14,c), name=None):
    return l.Input(shape=input_shape)
def c2(input_shape=(7,7,c), name=None):
    return l.Input(shape=input_shape)
def c3(input_shape=(4,4,c), name=None):
    return l.Input(shape=input_shape)
def c4(input_shape=(2,2,c), name=None):
    return l.Input(shape=input_shape)

def conv(f, k=3, s=1, p='same', act=None, b=True, k_init="he_normal", out_p=None, k_reg=0.00005, k_const=None, rank=2, name=None):
    # weight_decay = 0.00005
    if rank==3:
        conv_layer = l.Conv3D
    elif rank==2:
        conv_layer = l.Conv2D
    elif rank==1:
        conv_layer = l.Conv1D
    else:
        raise Exception("Conv Rank Error!!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    return conv_layer(filters=f, kernel_size=k, strides=s, padding=p, activation=act, use_bias=b, kernel_initializer=k_init,
                      bias_initializer='zeros', kernel_regularizer=k_reg, kernel_constraint=k_const, name=name)


def conv_transpose(f, k=2, s=1, p='valid', out_p='auto', act=None, b=True, k_init="he_normal", k_reg=0.00005,
                   k_const=None, rank=2, name=None):
    if rank==3:
        conv_transpose_layer = l.Conv3DTranspose
    elif rank==2:
        conv_transpose_layer = l.Conv2DTranspose
    else:
        raise Exception("Conv Transpose Rank Error!!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    if out_p=="auto":
        out_p = (W%s, H%s)

    return conv_transpose_layer(filters=f, kernel_size=k, strides=s, padding=p, output_padding=out_p,
                                activation=act, use_bias=b, kernel_initializer=k_init, bias_initializer='zeros',
                                kernel_regularizer=k_reg, kernel_constraint=k_const, name=name)

def relu(name=None):
    return l.ReLU()

def leakyrelu(name=None):
    return l.LeakyReLU()

def tanh(x, name=None):
    return keras.activations.tanh(x)

def batch_norm(m=0.99, e=1e-3, name=None):
    return l.BatchNormalization(momentum=m, epsilon=e)

def concat(axis=-1, name=None):
    return l.Concatenate(axis=axis)

def flatten(name=None):
    return l.Flatten()

def maxpool(k=2, s=2, p="valid", rank=2, name=None):
    if rank==3:
        maxpool_layer = l.MaxPooling3D
    elif rank==2:
        maxpool_layer = l.MaxPooling2D
    elif rank==1:
        maxpool_layer = l.MaxPooling1D
    else:
        raise Exception("Maxpool Rank Error!!")
    return maxpool_layer(pool_size=k, strides=s, padding=p)

def global_avgpool(rank=2, name=None):
    if rank==3:
        global_avgpool_layer = l.GlobalAveragePooling3D
    elif rank==2:
        global_avgpool_layer = l.GlobalAveragePooling2D
    elif rank==1:
        global_avgpool_layer = l.GlobalAveragePooling1D
    else:
        raise Exception("global_avgpool Rank Error!!")

    return global_avgpool_layer()

def upsample(rank=2, s=2, interpolation="bilinear", name=None):
    if rank==3:
        upsampling = l.UpSampling3D
    elif rank==2:
        upsampling = l.UpSampling2D
    elif rank==1:
        upsampling = l.UpSampling1D
    else:
        raise Exception("Upsample Rank Error!!")
    return upsampling(size=s, interpolation=interpolation)


def dense(f, act=None, b=True, k_init="he_normal", k_reg=0.00005, k_const=None, name=None):
    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const=tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0])

    return l.Dense(units=f, activation=act, use_bias=b, kernel_initializer=k_init,
                   bias_initializer="zeros", kernel_regularizer=k_reg, kernel_constraint=k_const)

def softmax(x, name=None):
    return l.Softmax()(x)

def dropout(rate=None, name=None):
    return l.Dropout(rate=rate)