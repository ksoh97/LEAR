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
Tr = Trainer()
Tr.train()