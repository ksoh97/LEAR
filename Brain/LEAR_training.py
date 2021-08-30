import os
import LEAR_layers as layers
import GPUtil
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tqdm

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices

l = keras.layers
K = keras.backend

file_name = "ResNet18"

ch = 64
fold = 5
epoch = 150
learning_rate = 0.0001
learning_decay = 0.98
batch_size = 12

class Utils:
    def __init__(self):
        self.project_path = '/home/ksoh/server_stratification/'
        self.data_path = '/DataRead/ksoh/js_ws_data/'

    def load_adni_data(self):
        """
        class #0: NC(433), class #1: pMCI(251), class #2: sMCI(497), class #3: AD(359)
        :return: NC, pMCI, sMCI, AD
        """
        dat = np.load(self.data_path + "total_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "labels.npy")
        return dat, lbl

    def data_permutation(self, lbl, cv):
        Total_NC_idx, Total_AD_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 3))
        Total_sMCI_idx, Total_pMCI_idx = np.squeeze(np.argwhere(lbl == 2)), np.squeeze(np.argwhere(lbl == 1))
        Total_MCI_idx = np.concatenate((Total_sMCI_idx, Total_pMCI_idx), axis=0)
        amount_NC, amount_MCI, amount_AD = int(len(Total_NC_idx) / fold), int(len(Total_MCI_idx) / fold), int(len(Total_AD_idx) / fold)

        NCvalid_idx = Total_NC_idx[cv * amount_NC:(cv + 1) * amount_NC]
        NCtrain_idx = np.setdiff1d(Total_NC_idx, NCvalid_idx)
        NCtest_idx = NCvalid_idx[:int(len(NCvalid_idx) / 2)]
        NCvalid_idx = np.setdiff1d(NCvalid_idx, NCtest_idx)

        MCIvalid_idx = Total_MCI_idx[cv * amount_MCI:(cv + 1) * amount_MCI]
        MCItrain_idx = np.setdiff1d(Total_MCI_idx, MCIvalid_idx)
        MCItest_idx = MCIvalid_idx[:int(len(MCIvalid_idx) / 2)]
        MCIvalid_idx = np.setdiff1d(MCIvalid_idx, MCItest_idx)

        ADvalid_idx = Total_AD_idx[cv * amount_AD:(cv + 1) * amount_AD]
        ADtrain_idx = np.setdiff1d(Total_AD_idx, ADvalid_idx)
        ADtest_idx = ADvalid_idx[:int(len(ADvalid_idx) / 2)]
        ADvalid_idx = np.setdiff1d(ADvalid_idx, ADtest_idx)

        self.train_all_idx = np.concatenate((NCtrain_idx, MCItrain_idx))
        self.train_all_idx = np.concatenate((self.train_all_idx, ADtrain_idx))
        self.valid_all_idx = np.concatenate((NCvalid_idx, MCIvalid_idx))
        self.valid_all_idx = np.concatenate((self.valid_all_idx, ADvalid_idx))
        self.test_all_idx = np.concatenate((NCtest_idx, MCItest_idx))
        self.test_all_idx = np.concatenate((self.test_all_idx, ADtest_idx))

        return self.train_all_idx, self.valid_all_idx, self.test_all_idx

    def seperate_data(self, data_idx, dat, lbl, CENTER=False):
        dat, lbl = dat[data_idx], lbl[data_idx]
        dat = np.squeeze(dat)
        lbl = np.where(lbl == 2, 1, lbl).astype("int32")
        lbl = np.where(lbl == 3, 2, lbl).astype("int32")
        lbl = np.eye(3)[lbl.squeeze()]
        lbl = lbl.astype('float32')
        if len(data_idx) == 1: dat = np.expand_dims(dat, axis=0)

        # Original
        if CENTER:
            for batch in range(len(data_idx)):
                # Quantile normalization
                Q1, Q3 = np.quantile(dat[batch], 0.1), np.quantile(dat[batch], 0.9)
                dat[batch] = np.where(dat[batch] < Q1, Q1, dat[batch])
                dat[batch] = np.where(dat[batch] > Q3, Q3, dat[batch])

                # Gaussian normalization
                m, std = np.mean(dat[batch]), np.std(dat[batch])
                dat[batch] = (dat[batch] - m) / std
            dat = np.expand_dims(dat, axis=-1)

        else:
            padding = 5
            npad = ((padding, padding), (padding, padding), (padding, padding))
            emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3]))

            for cnt, dat in enumerate(dat):
                tmp = np.pad(dat, npad, "constant")
                emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

            for batch in range(len(emp)):
                # Quantile normalization
                Q1, Q3 = np.quantile(emp[batch], 0.1), np.quantile(emp[batch], 0.9)
                emp[batch] = np.where(emp[batch] < Q1, Q1, emp[batch])
                emp[batch] = np.where(emp[batch] > Q3, Q3, emp[batch])

                # Gaussian normalization
                m, std = np.mean(emp[batch]), np.std(emp[batch])
                emp[batch] = (emp[batch] - m) / std
            dat = np.expand_dims(emp, axis=-1)

        return dat, lbl

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

class Trainer:
    def __init__(self):
        self.lr = learning_rate
        self.file_name = file_name
        self.decay = learning_decay
        self.epoch = epoch
        self.fold = fold
        self.batch_size = batch_size

        self.valid_acc, self.compare_acc, self.valid_loss, self.count = 0, 0, 0, 0
        tf.keras.backend.set_image_data_format("channels_last")
        self.valid_save, self.nii_save, self.model_select = False, False, False
        self.build_model()

        self.path = os.path.join("/DataCommon2/ksoh/classification_performance/3CLASS/" + self.file_name)
        if not os.path.exists(self.path): os.makedirs(self.path)

    def build_model(self):
        resnet = ResNet18()
        self.train_vars = []
        self.cls_model = resnet.build_model()
        self.train_vars += self.cls_model.trainable_variables

    def _train_one_batch(self, dat_all, lbl, gen_optim, train_vars, step, cv):
        with tf.GradientTape() as tape:
            res = self.cls_model({"All_in": dat_all}, training=True)["cls_out"]
            loss = K.mean(keras.losses.categorical_crossentropy(lbl, res))

        grads = tape.gradient(loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("%dfold_train_loss" % (cv + 1), loss, step=step)

    def _valid_logger(self, dat_all, lbl, epoch, cv):
        res = self.cls_model({"All_in": dat_all}, training=False)["cls_out"]

        valid_loss = K.mean(keras.losses.categorical_crossentropy(lbl, res))
        valid_acc = K.mean(K.equal(K.argmax(lbl, axis=-1), K.argmax(res, axis=-1)))

        self.valid_loss += valid_loss
        self.valid_acc += valid_acc
        self.count += 1

        if self.valid_save == True:
            self.valid_acc = self.valid_acc / self.count
            self.valid_loss = self.valid_loss / self.count

            if self.compare_acc <= self.valid_acc:
                self.model_select = True
                self.compare_acc = self.valid_acc

            elif self.valid_acc >= 0.55:
                self.model_select = True

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("%dfold_valid_loss" % (cv + 1), self.valid_loss, step=epoch)
                tf.summary.scalar("%dfold_valid_acc" % (cv + 1), self.valid_acc, step=epoch)
                self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                self.valid_save = False

    def train(self):
        util = Utils()
        dat, lbl = util.load_adni_data()
        long_path = "/DataCommon/ksoh/longitudinal/3class"
        long_nc, long_mci, long_ad = np.load(long_path+"/resized_quan_NC.npy"), np.load(long_path+"/resized_quan_MCI.npy"), np.load(long_path+"/resized_quan_AD.npy")
        long_nc, long_mci, long_ad = np.expand_dims(long_nc, axis=-1), np.expand_dims(long_mci, axis=-1), np.expand_dims(long_ad, axis=-1)
        long_lbl = np.append(np.zeros(len(long_nc)), np.ones(len(long_mci)))
        long_lbl = np.append(long_lbl, (np.ones(len(long_ad))+1)).astype("int32")

        for cv in range(0, self.fold):
            self.train_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid" % (cv + 1))
            self.test_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_test" % (cv + 1))
            self.train_all_idx, self.valid_all_idx, self.test_all_idx = util.data_permutation(lbl, cv)
            self.build_model()

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr, decay_steps=len(self.train_all_idx) // self.batch_size, decay_rate=self.decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)
            global_step, self.compare_acc = 0, 0

            for cur_epoch in tqdm.trange(self.epoch, desc="3class_resnet18_%s.py" % file_name):
                self.train_all_idx = np.random.permutation(self.train_all_idx)

                # training
                for cur_step in tqdm.trange(0, len(self.train_all_idx), self.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + self.batch_size]
                    cur_dat, cur_lbl = util.seperate_data(cur_idx, dat, lbl, CENTER=False)

                    self._train_one_batch(dat_all=cur_dat, lbl=cur_lbl, gen_optim=optim, train_vars=self.train_vars, step=global_step, cv=cv)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(self.valid_all_idx), self.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + self.batch_size]
                    val_dat, val_lbl = util.seperate_data(val_idx, dat, lbl, CENTER=True)

                    if val_step + self.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._valid_logger(dat_all=val_dat, lbl=val_lbl, epoch=cur_epoch, cv=cv)

                if self.model_select == True:
                    self.cls_model.save(os.path.join(self.path + '/%dfold_cls_model_%03d' % (cv + 1, cur_epoch)))
                    self.model_select = False

                # Test
                tot_true, tot_pred = 0, 0
                for tst_step in tqdm.trange(0, len(self.test_all_idx), self.batch_size, desc="Testing step: %dfold" % (cv + 1)):
                    tst_idx = self.test_all_idx[tst_step:tst_step + self.batch_size]
                    tst_dat, tst_lbl = util.seperate_data(tst_idx, dat, lbl, CENTER=True)
                    res = self.cls_model({"All_in": tst_dat}, training=False)["cls_out"]

                    if tst_step == 0:
                        tot_true, tot_pred = np.argmax(tst_lbl, axis=-1), np.argmax(res, axis=-1)
                    else:
                        tot_true = np.append(tot_true, np.argmax(tst_lbl, axis=-1))
                        tot_pred = np.append(tot_pred, np.argmax(res, axis=-1))

                acc = self.evaluation_matrics(tot_true, tot_pred)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_test_ACC" % (cv + 1), acc, step=cur_epoch)

                # Longitudinal test
                long_pred = self.cls_model(long_nc)["cls_out"]
                long_pred = np.concatenate((long_pred, self.cls_model(long_mci)["cls_out"]))
                long_pred = np.concatenate((long_pred, self.cls_model(long_ad)["cls_out"]))
                long_pred = np.argmax(long_pred, axis=-1)

                acc = self.evaluation_matrics(long_lbl, long_pred)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_long_test_ACC" % (cv + 1), acc, step=cur_epoch)

    def evaluation_matrics(self, y_true, y_pred):
        return K.mean(K.equal(y_true, y_pred))

    def MAUC(self, data, num_classes):
        """
        Calculates the MAUC over a set of multi-class probabilities and
        their labels. This is equation 7 in Hand and Till's 2001 paper.
        NB: The class labels should be in the set [0,n-1] where n = # of classes.
        The class probability should be at the index of its label in the
        probability list.
        I.e. With 3 classes the labels should be 0, 1, 2. The class probability
        for class '1' will be found in index 1 in the class probability list
        wrapped inside the zipped list with the labels.
        Args:
            data (list): A zipped list (NOT A GENERATOR) of the labels and the
                class probabilities in the form (m = # data instances):
                 [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
                  (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                                 ...
                  (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
                 ]
            num_classes (int): The number of classes in the dataset.
        Returns:
            The MAUC as a floating point value.
        """
        # Find all pairwise comparisons of labels
        import itertools as itertools
        class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]
        # Have to take average of A value with both classes acting as label 0 as this
        # gives different outputs for more than 2 classes
        sum_avals = 0
        for pairing in class_pairs:
            sum_avals += (self.a_value(data, zero_label=pairing[0], one_label=pairing[1]) + self.a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0
        return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7

    def a_value(self, probabilities, zero_label=0, one_label=1):
        """
        Approximates the AUC by the method described in Hand and Till 2001,
        equation 3.
        NB: The class labels should be in the set [0,n-1] where n = # of classes.
        The class probability should be at the index of its label in the
        probability list.
        I.e. With 3 classes the labels should be 0, 1, 2. The class probability
        for class '1' will be found in index 1 in the class probability list
        wrapped inside the zipped list with the labels.
        Args:
            probabilities (list): A zipped list of the labels and the
                class probabilities in the form (m = # data instances):
                 [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
                  (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                                 ...
                  (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
                 ]
            zero_label (optional, int): The label to use as the class '0'.
                Must be an integer, see above for details.
            one_label (optional, int): The label to use as the class '1'.
                Must be an integer, see above for details.
        Returns:
            The A-value as a floating point.
        """
        # Obtain a list of the probabilities for the specified zero label class
        expanded_points = []
        for instance in probabilities:
            if instance[0] == zero_label or instance[0] == one_label:
                expanded_points.append((instance[0].item(), instance[zero_label + 1].item()))
        sorted_ranks = sorted(expanded_points, key=lambda x: x[1])
        n0, n1, sum_ranks = 0, 0, 0
        # Iterate through ranks and increment counters for overall count and ranks of class 0
        for index, point in enumerate(sorted_ranks):
            if point[0] == zero_label:
                n0 += 1
                sum_ranks += index + 1  # Add 1 as ranks are one-based
            elif point[0] == one_label:
                n1 += 1
            else:
                pass  # Not interested in this class
        # print('Before: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
        if n0 == 0:
            n0 = 1e-10
        elif n1 == 0:
            n1 = 1e-10
        else:
            pass
        # print('After: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
        return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3

Tr = Trainer()
Tr.train()