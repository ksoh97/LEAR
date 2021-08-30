class ResNet18:
    def __init__(self, ch=ch):
        self.ch = ch
        self.input = layers.input_layer(input_shape=(96, 114, 96, 1), name="input")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act=None, n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def residual_block(self, x, ch, name):
        shortcut = x
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def residual_block_first(self, x, ch, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool%d" % name)
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut%d" % name)
            shortcut = shortcut(x)

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def build_model(self):
        enc_conv1_1 = self.conv_bn_act(x=self.input, f=ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
        max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

        # conv2_x
        enc_conv2_block1 = self.residual_block(x=max_pool1, ch=ch, name=2)
        enc_conv2_block2 = self.residual_block(x=enc_conv2_block1, ch=ch, name=3)

        # conv3_x
        enc_conv3_block1 = self.residual_block_first(x=enc_conv2_block2, ch=ch*2, strides=2, name=4)
        enc_conv3_block2 = self.residual_block(x=enc_conv3_block1, ch=ch*2, name=5)

        # conv4_x
        enc_conv4_block1 = self.residual_block_first(x=enc_conv3_block2, ch=ch*4, strides=2, name=6)
        enc_conv4_block2 = self.residual_block(x=enc_conv4_block1, ch=ch*4, name=7)

        # conv5_x
        enc_conv5_block1 = self.residual_block_first(x=enc_conv4_block2, ch=ch*8, strides=2, name=8)
        enc_conv5_block2 = self.residual_block(x=enc_conv5_block1, ch=ch*8, name=9)

        gap = layers.global_avgpool(rank=3, name="gap")(enc_conv5_block2)
        dense = self.dense_layer(x=gap, f=3, act=False, n="dense_2")
        cls_out = layers.softmax(x=dense, name="softmax")

        self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")
        return self.cls_model