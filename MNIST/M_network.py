import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_addons as tfa
from MNIST import M_config as conf, M_layers


class MNIST_network:
    def __init__(self, ch=conf.cfmap_ch):
        self.ch = ch
        self.layers = {}
        self.input = M_layers.input_layer(name="input")
        self.c1, self.c2, self.c3 = M_layers.c1(name="c1"), M_layers.c2(name="c2"), M_layers.c3(name="c3")
        self.pretraining_clf(), self.CFmap_generator()

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

        self.layers[n + "_conv"] = conv_l
        self.layers[n + "_norm"] = norm_l

        if act:
            act_l = M_layers.relu(name=n + "_relu")
            self.layers[n + "_relu"] = act_l
            out = act_l(out)
        return out

    def conv_bn_act_reuse(self, x, n, act=True):
        out = self.layers[n + "_conv"](x)
        out = self.layers[n + "_norm"](out)

        if act:
            out = self.layers[n + "_relu"](out)
        return out

    def concat(self, x, y, n):
        concat_l = M_layers.concat(name=n + "_concat")
        self.layers[n + "_concat"] = concat_l
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = M_layers.flatten(n + "_flatten")(x)
        self.layers[n + "_flatten"] = flatten_l
        return flatten_l

    def flatten_layer_reuse(self, x, n):
        return self.layers[n + "_flatten"](x)

    def dense_layer(self, x, f, act="relu", n=None):
        dense_l = M_layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)
        self.layers[n + "_dense"] = dense_l

        if act:
            act_l = M_layers.relu(n + "_relu")
            self.layers[n + "_relu"] = act_l
            out = act_l(out)
        return out

    def dense_layer_reuse(self, x, n, act=True):
        out = self.layers[n + "_dense"](x)
        if act: out = self.layers[n + "_relu"](out)
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

        cls_out = M_layers.softmax(x=dense_2, name="softmax")

        self.enc_model = keras.Model({"enc_in": self.input}, {"enc_out": enc_conv4_2}, name="enc_model")
        self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")

        return self.enc_model, self.cls_model

    def CFmap_generator(self):
        # Encoder
        enc_conv1_1 = self.conv_bn_act(x=self.input, k=3, s=1, f=self.ch, p="same", n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act(x=enc_conv1_1, k=4, s=2, f=self.ch, p="same", n="enc_conv1_2")

        enc_conv2_1 = self.conv_bn_act(x=enc_conv1_2, k=3, s=1, f=self.ch * 2, p="same", n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act(x=enc_conv2_1, k=4, s=2, f=self.ch * 2, p="same", n="enc_conv2_2")

        enc_conv3_1 = self.conv_bn_act(x=enc_conv2_2, k=3, s=1, f=self.ch * 4, p="same", n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act(x=enc_conv3_1, k=4, s=2, f=self.ch * 4, p="same", n="enc_conv3_2")

        enc_conv4_1 = self.conv_bn_act(x=enc_conv3_2, k=3, s=1, f=self.ch * 8, p="same", n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act(x=enc_conv4_1, k=4, s=2, f=self.ch * 8, p="same", n="enc_conv4_2")

        # Decoder
        dec_up3 = M_layers.upsample(rank=2, name="dec_up3")(enc_conv4_2)
        dec_upconv3 = self.conv_bn_act(dec_up3, f=self.ch * 4, k=3, s=1, p="same", n="dec_upconv3")
        dec_code_concat3 = self.concat(enc_conv3_2, self.c3, n="dec_code_concat3")
        dec_code_conv3 = self.conv_bn_act(x=dec_code_concat3, f=self.ch * 4, k=3, s=1, p="same", n="dec_code_conv3")
        dec_concat3 = self.concat(dec_code_conv3, dec_upconv3, n="dec_concat3")
        dec_conv3 = self.conv_bn_act(dec_concat3, f=self.ch * 4, k=3, s=1, p="same", n="dec_conv3")

        dec_up2 = M_layers.upsample(rank=2, name="dec_up2")(dec_conv3)
        dec_upconv2 = self.conv_bn_act(dec_up2, f=self.ch * 2, k=2, s=1, p="valid", n="dec_upconv2")
        dec_code_concat2 = self.concat(enc_conv2_2, self.c2, n="dec_code_concat2")
        dec_code_conv2 = self.conv_bn_act(x=dec_code_concat2, f=self.ch * 2, k=3, s=1, p="same",
                                          n="dec_code_conv2")
        dec_concat2 = self.concat(dec_code_conv2, dec_upconv2, n="dec_concat2")
        dec_conv2 = self.conv_bn_act(dec_concat2, f=self.ch * 2, k=3, s=1, p="same", n="dec_conv2")

        dec_up1 = M_layers.upsample(rank=2, name="dec_up1")(dec_conv2)
        dec_upconv1 = self.conv_bn_act(dec_up1, f=self.ch, k=3, s=1, p="same", n="dec_upconv1")
        dec_code_concat1 = self.concat(enc_conv1_2, self.c1, n="dec_code_concat1")
        dec_code_conv1 = self.conv_bn_act(x=dec_code_concat1, f=self.ch, k=3, s=1, p="same", n="dec_code_conv1")
        dec_concat1 = self.concat(dec_code_conv1, dec_upconv1, n="dec_concat1")
        dec_conv1 = self.conv_bn_act(dec_concat1, f=self.ch, k=3, s=1, p="same", n="dec_conv1")

        dec_up = self.conv_bn_act(dec_conv1, f=1, k=4, s=2, p="same", act=False, trans=True, n="dec_up")
        dec_out = M_layers.tanh(x=dec_up, name="dec_out_tanh")

        self.dec_model = keras.Model({"dec_in": self.input, "c1": self.c1, "c2": self.c2, "c3": self.c3},
                                     {"dec_out": dec_out}, name="dec_model")

        # Classifier
        enc_conv1_1 = self.conv_bn_act_reuse(x=self.input, n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act_reuse(x=enc_conv1_1, n="enc_conv1_2")

        enc_conv2_1 = self.conv_bn_act_reuse(x=enc_conv1_2, n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act_reuse(x=enc_conv2_1, n="enc_conv2_2")

        enc_conv3_1 = self.conv_bn_act_reuse(x=enc_conv2_2, n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act_reuse(x=enc_conv3_1, n="enc_conv3_2")

        enc_conv4_1 = self.conv_bn_act_reuse(x=enc_conv3_2, n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act_reuse(x=enc_conv4_1, n='enc_conv4_2')

        fc1 = self.flatten_layer(x=enc_conv4_2, n="enc_flatten")
        drop5 = M_layers.dropout(rate=0.5, name="enc_drop5")(fc1)
        dense_1 = self.dense_layer(x=drop5, f=128, n="dense_1")
        drop6 = M_layers.dropout(rate=0.25, name="enc_drop6")(dense_1)
        dense_2 = self.dense_layer(x=drop6, f=10, act=None, n="dense_3")

        cls_out = M_layers.softmax(x=dense_2, name="softmax")

        self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")

        return self.cls_model, self.dec_model

class Discriminator:
    def __init__(self, ch=conf.disc_ch):
        self.ch = ch
        self.discri_input = M_layers.input_layer(name="discri_input")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", batch=True, act=True):
        c_layer = M_layers.conv
        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")

        out = conv_l(x)

        if batch:
            batch_l = M_layers.batch_norm(name=n + "_batchnorm")
            out = batch_l(out)

        if act:
            act_l = M_layers.leakyrelu(name=n + "_leakyrelu")
            out = act_l(out)
        return out

    def flatten_layer(self, x, n=None):
        flatten_l = M_layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act="leakyrelu", n=None):
        dense_l = M_layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = M_layers.leakyrelu(n + "_leakyrelu")
            out = act_l(out)

        return out

    def build_model(self):
        discri_conv1_1 = self.conv_bn_act(x=self.discri_input, k=3, s=1, p="same", f=self.ch, batch=False, n="discri_conv1_1")
        discri_conv1_2 = self.conv_bn_act(x=discri_conv1_1, k=4, s=2, p="same", f=self.ch, n="discri_conv1_2")

        discri_conv2_1 = self.conv_bn_act(x=discri_conv1_2, k=3, s=1, p="same", f=self.ch * 2, n="discri_conv2_1")
        discri_conv2_2 = self.conv_bn_act(x=discri_conv2_1, k=4, s=2, p="same", f=self.ch * 2, n="discri_conv2_2")

        discri_conv3_1 = self.conv_bn_act(x=discri_conv2_2, k=3, s=1, p="same", f=self.ch * 4, n="discri_conv3_1")
        discri_conv3_2 = self.conv_bn_act(x=discri_conv3_1, k=4, s=2, p="same", f=self.ch * 4, n="discri_conv3_2")

        discri_conv4_1 = self.conv_bn_act(x=discri_conv3_2, k=3, s=1, p="same", f=self.ch * 8, n="discri_conv4_1")
        discri_conv4_2 = self.conv_bn_act(x=discri_conv4_1, k=4, s=2, p="same", f=self.ch * 8, n="discri_conv4_2")

        fc1 = self.flatten_layer(x=discri_conv4_2, n="discri_flatten")
        logit = self.dense_layer(x=fc1, f=1, act=None, n="discri_logit")
        output = tf.identity(logit)

        self.discriminator_model = keras.Model({"discri_in": self.discri_input}, {"discri_out": output},
                                               name="discriminator_model")
        return self.discriminator_model

# TODO: Change the discriminator network using a spectral normalization
# class Discriminator:
#     def __init__(self, ch=conf.disc_ch):
#         self.ch = ch
#         self.discri_input = M_layers.input_layer(name="discri_input")
#         self.build_model()
#
#     def conv_sn_act(self, x, f, n, s=1, k=None, p="same", batch=True, act=True):
#         c_layer = M_layers.conv
#         if k:
#             conv_l = c_layer(f=f, p=p, k=k, s=s, name=n + "_conv")
#         else:
#             conv_l = c_layer(f=f, p=p, name=n + "_conv")
#
#         if batch:
#             out = tfa.layers.SpectralNormalization(conv_l, name=n+"_norm")(x)
#         else:
#             out = conv_l(x)
#         if act:
#             act_l = M_layers.leakyrelu(name=n + "_leakyrelu")
#             out = act_l(out)
#         return out
#
#     def flatten_layer(self, x, n=None):
#         flatten_l = M_layers.flatten(n + "_flatten")(x)
#         return flatten_l
#
#     def dense_layer(self, x, f, n=None):
#         dense_l = M_layers.dense(f, act=None, name=n + "_dense")
#         out = dense_l(x)
#         return out
#
#     def build_model(self):
#         discri_conv1_1 = self.conv_sn_act(x=self.discri_input, k=3, s=1, p="same", f=self.ch, batch=False, n="discri_conv1_1")
#         discri_conv1_2 = self.conv_sn_act(x=discri_conv1_1, k=4, s=2, p="same", f=self.ch, n="discri_conv1_2")
#
#         discri_conv2_1 = self.conv_sn_act(x=discri_conv1_2, k=3, s=1, p="same", f=self.ch * 2, n="discri_conv2_1")
#         discri_conv2_2 = self.conv_sn_act(x=discri_conv2_1, k=4, s=2, p="same", f=self.ch * 2, n="discri_conv2_2")
#
#         discri_conv3_1 = self.conv_sn_act(x=discri_conv2_2, k=3, s=1, p="same", f=self.ch * 4, n="discri_conv3_1")
#         discri_conv3_2 = self.conv_sn_act(x=discri_conv3_1, k=4, s=2, p="same", f=self.ch * 4, n="discri_conv3_2")
#
#         discri_conv4_1 = self.conv_sn_act(x=discri_conv3_2, k=3, s=1, p="same", f=self.ch * 8, n="discri_conv4_1")
#         discri_conv4_2 = self.conv_sn_act(x=discri_conv4_1, k=4, s=2, p="same", f=self.ch * 8, n="discri_conv4_2")
#
#         flatten = self.flatten_layer(x=discri_conv4_2, n="flatten")
#         dense = self.dense_layer(x=flatten, f=1, n="dense")
#         logit = tf.identity(dense)
#
#         self.discriminator_model = keras.Model({"discri_in": self.discri_input}, {"discri_out": logit},
#                                                name="discriminator_model")
#         return self.discriminator_model