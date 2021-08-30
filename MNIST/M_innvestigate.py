import os
import pickle
from MNIST import M_config as conf, M_layers
import numpy as np

os.environ["GOOOOG"] = "1"
import tensorflow as tf
# if os.environ["GOOOOG"]=="0": import tensorflow.keras as keras
# else: import keras

from keras.datasets import mnist
def fetch_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train.astype("float32")/255.
    x_test = x_test.astype("float32")/255.

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = fetch_data()

y_test = np.eye(10)[y_test]

class MNIST_network:
    def __init__(self, ch=conf.cfmap_ch):
        self.ch = ch
        self.input = M_layers.input_layer(name="input")
        self.c1, self.c2, self.c3 = M_layers.c1(name="c1"), M_layers.c2(name="c2"), M_layers.c3(name="c3")
        self.pretraining_clf()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", out_p=None, act=True, trans=False):
        if trans:
            c_layer = M_layers.conv_transpose
        else:
            c_layer = M_layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")

        out = conv_l(x)
        norm_l = M_layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = M_layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def flatten_layer(self, x, n=None):
        flatten_l = M_layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act="relu", n=None):
        dense_l = M_layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = M_layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def pretraining_clf(self):
        enc_conv1_1 = self.conv_bn_act(x=self.input, k=3, s=1, f=self.ch, p="same", n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act(x=enc_conv1_1, k=4, s=2, f=self.ch, p="same", n="enc_conv1_2")

        enc_conv2_1 = self.conv_bn_act(x=enc_conv1_2, k=3, s=1, f=self.ch * 2, p="same", n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act(x=enc_conv2_1, k=4, s=2, f=self.ch * 2, p="same", n="enc_conv2_2")

        enc_conv3_1 = self.conv_bn_act(x=enc_conv2_2, k=3, s=1, f=self.ch * 4, p="same", n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act(x=enc_conv3_1, k=4, s=2, f=self.ch * 4, p="same", n="enc_conv3_2")

        enc_conv4_1 = self.conv_bn_act(x=enc_conv3_2, k=3, s=1, f=self.ch * 8, p="same", n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act(x=enc_conv4_1, k=4, s=2, f=self.ch * 8, p="same", n="enc_conv4_2")

        fc1 = self.flatten_layer(x=enc_conv4_2, n="enc_flatten")
        drop5 = M_layers.dropout(rate=0.5, name="enc_drop5")(fc1)
        dense_1 = self.dense_layer(x=drop5, f=128, n="dense_1")
        drop6 = M_layers.dropout(rate=0.25, name="enc_drop6")(dense_1)
        dense_2 = self.dense_layer(x=drop6, f=10, act=None, n="dense_3")

        self.cls_model = keras.Model(self.input, dense_2, name="cls_model")

        return self.cls_model

def get_model():
    model = MNIST_network().cls_model
    # Weight saving..
    # weight_dict ={}
    # for a in cls_model.variables:
    #     weight_dict[a.name] = a.numpy()
    # with open("/home/ksoh/submit_CVPR/mnist_model_weights.pkl", "wb") as f:
    #     pickle.dump(weight_dict, f)

    with open("/home/ksoh/submit_CVPR/mnist_model_weights.pkl", "rb") as f:
        weight_dict = pickle.load(f)

    new_weight_dict = {}
    for fff in weight_dict:
        old_f = fff
        if fff[0] == "b":
            if fff.startswith("batch_normalization/"):
                fff = fff.replace("batch_normalization/", "batch_normalization_1/")
            else:
                fff = "batch_normalization_%d/%s" % (
                    int(fff.split("_")[2].split("/")[0]) + 1, fff.split("/")[-1])
        elif fff[0] == "d":
            if fff.startswith("dense/"):
                fff = fff.replace("dense/", "dense_1/")
            else:
                fff = "dense_%d/%s" % (int(fff.split("_")[1].split("/")[0]) + 1, fff.split("/")[-1])

        if os.environ["GOOOOG"] == "0":
            new_weight_dict[old_f] = weight_dict[old_f]
        else:
            new_weight_dict[fff] = weight_dict[old_f]

    for l in model.layers:
        for old_w in l.weights:
            tf.keras.backend.set_value(old_w, new_weight_dict[old_w.name])
    return model

model = get_model()