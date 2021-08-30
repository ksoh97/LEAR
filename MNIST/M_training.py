import tensorflow as tf
import tensorflow.keras as keras
from MNIST import M_utils, M_network as net, M_losses, M_config as conf, M_test
import numpy as np
import GPUtil
import tqdm
import os

K = keras.backend

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices

file_name = conf.file_name
mode = conf.mode

if mode == 0:
    epoch = conf.epoch
    batch_size = conf.batch_size
    learning_rate = conf.lr
    learning_decay = conf.lr_decay
else:
    epoch = conf.epoch
    batch_size = conf.batch_size
    generator_step, discriminator_step = conf.g_step, conf.d_step
    learning_rate_g, learning_rate_d = conf.lr_g, conf.lr_d
    learning_decay = conf.lr_decay
    smooth = conf.one_sided_label_smoothing
    beta_1 = conf.beta_1
    loss_type = conf.loss_type

class Trainer:
    def __init__(self):
        if mode == 0:
            self.save_path = os.path.join(conf.save_path + file_name)
        else:
            self.save_path = os.path.join(conf.save_path + file_name)

        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.valid_acc, self.valid_loss, self.compare_acc, self.count = 0, 0, 0, 0
        self.model_select, self.valid_save, self.plt_save = False, False, False
        self.build_model()

    def build_model(self):
        if mode == 0:
            network = net.MNIST_network()
            self.train_vars = []
            self.enc_model, self.cls_model = network.pretraining_clf()
            self.train_vars += self.cls_model.trainable_variables

        elif mode == 1:
            g_network = net.MNIST_network()
            d_network = net.Discriminator()
            self.train_vars = []
            self.train_discri_vars = []

            self.discriminator_model = d_network.build_model()
            self.train_discri_vars += self.discriminator_model.trainable_variables

            self.cls_model, self.dec_model = g_network.CFmap_generator()
            cls_load_weights = conf.cls_weight_path
            self.cls_model.load_weights(cls_load_weights)
            for layer in self.cls_model.layers:
                layer.trainable = False

            enc_load_weights = conf.enc_weight_path
            self.dec_model.load_weights(enc_load_weights)
            save_variables = False

            for variables in self.dec_model.trainable_variables:
                if "dec" in variables.name:
                    save_variables = True
                if save_variables:
                    self.train_vars += [variables]

    def cycle_consistency(self, pseudo_image, lbl):
        c1, c2, c3 = M_utils.codemap(target_c=lbl)
        tilde_map = self.dec_model({"enc_in": pseudo_image, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
        like_input = pseudo_image + tilde_map
        return like_input

    def train_one_batch(self, train_dat, train_lbl, optim, disc_optim, train_vars, train_discri_vars, step, target_c):
        if mode == 0:
            with tf.GradientTape() as tape:
                res = self.cls_model({"cls_in": train_dat}, training=True)["cls_out"]
                loss = M_losses.CE_loss(train_lbl, res)

            grads = tape.gradient(loss, train_vars)
            optim.apply_gradients(zip(grads, train_vars))

            if step % 20 == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("mode0_train_loss", loss, step=step)

        elif mode == 1:
            pred_lbl = self.cls_model({"cls_in": train_dat}, training=False)["cls_out"]
            c1, c2, c3 = M_utils.codemap(target_c=target_c)
            # Discriminator step
            if step % discriminator_step == 0 or step == 0:
                with tf.GradientTape() as tape:
                    CFmap = self.dec_model({"enc_in": train_dat, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
                    pseudo_image = train_dat + CFmap

                    real = self.discriminator_model({"discri_in": train_dat}, training=True)["discri_out"]
                    fake = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]
                    total_loss = loss_type["dis"] * M_losses.discriminator_loss(real, fake)

                grads = tape.gradient(total_loss, train_discri_vars)
                disc_optim.apply_gradients(zip(grads, train_discri_vars))

                if step % 10 == 0:
                    with self.discriminator_summary_writer.as_default():
                        tf.summary.scalar("discriminator_loss", total_loss, step=step)

            # Generator step
            if step % generator_step == 0 or step == 0:
                with tf.GradientTape() as tape:
                    CFmap = self.dec_model({"enc_in": train_dat, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
                    pseudo_image = train_dat + CFmap

                    like_input = self.cycle_consistency(pseudo_image, pred_lbl)
                    discri_res = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]
                    res = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

                    if smooth:
                        cls = loss_type["cls"] * M_losses.CE_loss(M_losses.one_sided_label_smoothing(target_c), res)
                    else:
                        cls = loss_type["cls"] * M_losses.CE_loss(target_c, res)

                    gan = loss_type["GAN"] * M_losses.generator_loss(target_c, discri_res)
                    cyc = loss_type["cyc"] * M_losses.cycle_loss(train_dat, like_input)
                    l1_norm = loss_type["norm"] * M_losses.L1_norm(CFmap)
                    total_loss = cls + cyc + gan + l1_norm

                grads = tape.gradient(total_loss, train_vars)
                optim.apply_gradients(zip(grads, train_vars))

                if step % 10 == 0:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar("generator_total_train_loss", total_loss, step=step)
                    with self.generator_summary_writer.as_default():
                        tf.summary.scalar("generator_cls_loss", cls, step=step)
                        tf.summary.scalar("generator_gan_loss", gan, step=step)
                        tf.summary.scalar("generator_cyc_loss", cyc, step=step)
                        tf.summary.scalar("generator_l1_loss", l1_norm, step=step)

    def valid_logger(self, valid_dat, valid_lbl, epoch, target_c):
        if mode == 0:
            res = self.cls_model({"cls_in": valid_dat}, training=False)["cls_out"]
            valid_loss = M_losses.CE_loss(valid_lbl, res)
            valid_acc = M_losses.acc(valid_lbl, res)

            self.valid_loss += valid_loss
            self.valid_acc += valid_acc
            self.count += 1

            if self.valid_save == True:
                self.valid_acc /= self.count
                self.valid_loss /= self.count

                if self.compare_acc <= self.valid_acc:
                    self.compare_acc = self.valid_acc
                    self.model_select = True
                    print("Valid ACC: %f" % self.compare_acc)

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("mode0_valid_loss", self.valid_loss, step=epoch)
                    tf.summary.scalar("mode0_valid_acc", self.valid_acc, step=epoch)
                    self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                    self.valid_save = False

        elif mode == 1:
            pred_lbl = self.cls_model({"cls_in": valid_dat}, training=False)["cls_out"]
            c1, c2, c3 = M_utils.codemap(target_c=target_c)
            CFmap = self.dec_model({"enc_in": valid_dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
            pseudo_image = valid_dat + CFmap

            like_input = self.cycle_consistency(pseudo_image, pred_lbl)
            discri_res = self.discriminator_model({"discri_in": pseudo_image}, training=False)["discri_out"]
            res = self.cls_model({"effect_in": pseudo_image}, training=False)["cls_out"]

            if smooth:
                cls = loss_type["cls"] * M_losses.CE_loss(M_losses.one_sided_label_smoothing(target_c), res)
            else:
                cls = loss_type["cls"] * M_losses.CE_loss(target_c, res)

            gan = loss_type["GAN"] * M_losses.generator_loss(target_c, discri_res)
            cyc = loss_type["cyc"] * M_losses.cycle_loss(valid_dat, like_input)
            l1_norm = loss_type["norm"] * M_losses.L1_norm(CFmap)

            valid_loss = cls + cyc + gan + l1_norm
            valid_acc = M_losses.acc(target_c, res)

            self.valid_loss += valid_loss
            self.valid_acc += valid_acc
            self.count += 1

            if self.valid_save:
                self.valid_loss = self.valid_loss / self.count
                self.valid_acc = self.valid_acc / self.count
                print("Epoch:%03d, valid ACC: %f" % (epoch, self.valid_acc))

                if self.compare_acc <= self.valid_acc:
                    self.compare_acc = self.valid_acc
                    self.model_select = True

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("mode1_valid_loss", self.valid_loss, step=epoch)
                    tf.summary.scalar("mode1_valid_acc", self.valid_acc, step=epoch)
                    self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                self.valid_save = False

    def train(self):
        images_train, labels_train, images_test, labels_test = M_utils.data_load()
        self.global_step = 0

        self.train_summary_writer = tf.summary.create_file_writer(self.save_path + "/train")
        self.valid_summary_writer = tf.summary.create_file_writer(self.save_path + "/valid")
        self.test_summary_writer = tf.summary.create_file_writer(self.save_path + "/test")
        self.all_train_idx, self.all_valid_idx, self.all_test_idx = M_utils.permutation(labels_train, labels_test)
        self.build_model()

        if mode == 0:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=len(self.all_train_idx) // batch_size, decay_rate=learning_decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)

            for cur_epoch in tqdm.trange(0, epoch, desc=file_name):
                self.all_train_idx = np.random.permutation(self.all_train_idx)

                # Training step
                for trn_step in tqdm.trange(0, len(self.all_train_idx), batch_size, desc="Training a model: epoch%d" % cur_epoch):
                    train_idx = self.all_train_idx[trn_step:trn_step + batch_size]
                    train_dat, train_lbl = M_utils.seperate_data(train_idx, images_train, labels_train, CENTER=False)
                    self.train_one_batch(train_dat=train_dat, train_lbl=train_lbl, optim=optim, disc_optim=None,
                                         train_vars=self.train_vars,
                                         train_discri_vars=None, step=self.global_step, target_c=None)
                    self.global_step += 1

                # Validation step
                for val_step in tqdm.trange(0, len(self.all_valid_idx), batch_size, desc="Validation"):
                    valid_idx = self.all_valid_idx[val_step:val_step + batch_size]
                    valid_dat, valid_lbl = M_utils.seperate_data(valid_idx, images_train, labels_train, CENTER=True)

                if val_step + batch_size >= len(valid_dat): self.valid_save = True
                self.valid_logger(valid_dat=valid_dat, valid_lbl=valid_lbl, epoch=cur_epoch, target_c=None)

                if self.model_select:
                    self.cls_model.save(self.save_path + "/cls_model_%03d" % cur_epoch)
                    self.enc_model.save(self.save_path + "/enc_model_%03d" % cur_epoch)
                    self.model_select = False

                # Testing step
                self.total_tst_acc, self.total_tst_loss, self.tst_count = 0, 0, 0
                for tst_step in tqdm.trange(0, len(self.all_test_idx), batch_size, desc="Test"):
                    test_idx = self.all_test_idx[tst_step:tst_step + batch_size]
                    test_dat, test_lbl = M_utils.seperate_data(test_idx, images_test, labels_test, CENTER=True)

                    res = self.cls_model({"cls_in": test_dat}, training=False)["cls_out"]
                    test_loss = M_losses.CE_loss(test_lbl, res)
                    test_acc = M_losses.acc(test_lbl, res)

                    self.total_tst_acc += test_acc
                    self.total_tst_loss += test_loss
                    self.tst_count += 1

                print("Test ACC: %f" % (self.total_tst_acc / self.tst_count))
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("mode0_test_loss", (self.total_tst_loss / self.tst_count), step=cur_epoch)
                    tf.summary.scalar("mode0_test_acc", (self.total_tst_acc / self.tst_count), step=cur_epoch)

        elif mode == 1:
            self.discriminator_summary_writer = tf.summary.create_file_writer(self.save_path + "/train_discriminator")
            self.generator_summary_writer = tf.summary.create_file_writer(self.save_path + "/train_generator")

            g_lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate_g, decay_steps=len(self.all_train_idx) // batch_size, decay_rate=learning_decay, staircase=True)
            d_lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate_d, decay_steps=len(self.all_train_idx) // batch_size, decay_rate=learning_decay, staircase=True)
            gen_optim = keras.optimizers.Adam(learning_rate=g_lr_schedule, beta_1=beta_1)
            disc_optim = keras.optimizers.Adam(learning_rate=d_lr_schedule, beta_1=beta_1)

            for cur_epoch in tqdm.trange(0, epoch, desc=file_name):
                train_idx = np.squeeze(np.random.permutation(self.all_train_idx))

                # Training step
                for cur_step in tqdm.trange(0, len(train_idx), batch_size, desc="Epoch : %d" % cur_epoch):
                    cur_idx = train_idx[cur_step:cur_step + batch_size]
                    cur_dat, cur_lbl = M_utils.seperate_data(cur_idx, images_train, labels_train, CENTER=False)
                    target_c = M_utils.code_creator(size=cur_dat.shape[0], MINI=True)

                    self.train_one_batch(train_dat=cur_dat, train_lbl=cur_lbl, optim=gen_optim, disc_optim=disc_optim, train_vars=self.train_vars,
                                         train_discri_vars=self.train_discri_vars, target_c=target_c, step=self.global_step)
                    self.global_step += 1

                # validation step
                for val_step in tqdm.trange(0, len(self.all_valid_idx), batch_size, desc="Validation step"):
                    val_idx = np.squeeze(self.all_valid_idx[val_step:val_step + batch_size])
                    val_dat, val_lbl = M_utils.seperate_data(val_idx, images_train, labels_train, CENTER=True)
                    target_c = M_utils.code_creator(size=val_dat.shape[0], MINI=True)

                    if val_step == ((len(self.all_valid_idx) // batch_size) - 1) * batch_size: self.valid_save = True
                    self.valid_logger(valid_dat=val_dat, valid_lbl=val_lbl, epoch=cur_epoch, target_c=target_c)

                if self.model_select == True:
                    self.dec_model.save(os.path.join(self.save_path + '/Model/dec_model_%03d' % (cur_epoch)))
                    self.model_select = False

                # Test step
                self.total_tst_acc, self.tst_count = 0, 0
                for tst_step in tqdm.trange(0, len(self.all_test_idx), batch_size, desc="Test step"):
                    tst_idx = np.squeeze(self.all_test_idx[tst_step:tst_step + batch_size])
                    tst_dat, tst_lbl = M_utils.seperate_data(tst_idx, images_test, labels_test, CENTER=True)
                    target_c = M_utils.code_creator(size=tst_dat.shape[0], MINI=True)

                    c1, c2, c3 = M_utils.codemap(target_c=target_c)

                    CFmap = self.dec_model({"dec_in": tst_dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]

                    pseudo_image = tst_dat + CFmap

                    res = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]
                    tst_acc = M_losses.acc(target_c, res)
                    self.total_tst_acc += tst_acc
                    self.tst_count += 1

                print("Test ACC: %f" % (self.total_tst_acc / self.tst_count))
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("mode1_test_acc", (self.total_tst_acc / self.tst_count), step=cur_epoch)

                # Counterfactual map visualization
                M_test.visualization(self.save_path, images_test, self.dec_model, cur_epoch)

    def aa(self):
        images_train, labels_train, images_test, labels_test = M_utils.data_load()
        M_test.visualization(save_path=self.save_path, images_test=images_test, dec_model=self.dec_model)

tr = Trainer()
tr.train()